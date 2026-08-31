"""
[SCI 보강용 추가 실험] Pruning + KD 결합.

기존 실험은 "YOLOv8l(teacher) -> YOLOv8n(student)"처럼 capacity gap(14.5배)이 매우 커서
KD가 실패했다는 결과였음. 이 스크립트는 반대로 capacity gap이 아주 작은 상황
("원본 미프루닝 YOLOv8n" -> "45% 프루닝된 YOLOv8n")에서 KD를 다시 검증함.

가설: capacity gap이 작으면 KD가 프루닝만 단독으로 했을 때보다 정확도 회복에
도움이 될 것 (프루닝으로 손실된 정보를 원본 미프루닝 모델이 보완).
-> 이 실험이 성공하면 "capacity gap 크기가 KD 성패의 핵심 변수"라는 논문의 핵심
   주장을 훨씬 강하게 뒷받침하는 대조 실험(ablation)이 되어 SCI 심사에서 노벨티로
   내세울 수 있음. 실패해도 "구조적 프루닝이 더 안정적"이라는 기존 결론이 더 강화됨.

방법: finetune_pruned.py와 동일한 학습 루프 + feature-level distillation loss
      (teacher/student 백본 마지막 feature map 간 MSE)를 detection loss에 추가.

사용:
    python distill_pruned.py \
        --teacher-weights ../outputs/yolov8n_baseline/weights/best.pt \
        --pruned-weights ../outputs/yolov8n_pruned_r45/weights/pruned.pt \
        --data ../configs/bdd100k.yaml \
        --epochs 100 --batch 64 --kd-weight 0.5 --name yolov8n_pruned_r45_kd
"""
import argparse
import os

import torch
import torch.nn.functional as F
from ultralytics import YOLO

import c2f_v2_utils  # noqa: F401  (pruned.pt unpickle에 C2f_v2 정의 필요)


def atomic_save(yolo_obj, path):
    tmp_path = path + ".tmp"
    try:
        yolo_obj.save(tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"  [경고] 저장 실패 ({path}): {e}. 이전 체크포인트는 그대로 유지됩니다.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def register_feature_hook(model, layer_index=-1):
    """model.model[layer_index] (SPPF 등 백본 마지막 레이어)의 출력을 캡처하는 훅."""
    features = {}

    def hook(_module, _inp, out):
        features["feat"] = out

    target = model.model[layer_index]
    target.register_forward_hook(hook)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-weights", type=str, required=True,
                         help="원본 미프루닝 baseline 가중치 (student와 같은 YOLOv8n 계열)")
    parser.add_argument("--pruned-weights", type=str, required=True,
                         help="프루닝 직후(파인튜닝 전) 가중치")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kd-weight", type=float, default=0.5,
                         help="detection loss 대비 feature KD loss 가중치")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--project", type=str, default="../outputs")
    args = parser.parse_args()

    weights_dir = os.path.join(args.project, args.name, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    print(">>> teacher(원본 미프루닝) / student(프루닝됨) 로드 중...")
    teacher_yolo = YOLO(args.teacher_weights)
    teacher = teacher_yolo.model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_yolo = YOLO(args.pruned_weights)
    student = student_yolo.model.to(device)
    student.train()

    teacher_feat = register_feature_hook(teacher, layer_index=-2)
    student_feat = register_feature_hook(student, layer_index=-2)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    _use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

    from ultralytics.models.yolo.detect import DetectionTrainer
    trainer_args = dict(model=args.pruned_weights, data=args.data, epochs=1,
                         batch=args.batch, imgsz=args.imgsz, device=args.device, verbose=False)
    trainer = DetectionTrainer(overrides=trainer_args)
    trainer._setup_train(world_size=1)
    train_loader = trainer.train_loader

    # model.loss()가 self.args.box/.cls/.dfl 등을 참조하므로, 체크포인트의 dict형 args 대신
    # trainer의 SimpleNamespace 기반 args로 덮어써야 함 (finetune_pruned.py와 동일한 이슈).
    student.args = trainer.args

    best_map50 = 0.0
    no_improve_count = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_det_loss, epoch_kd_loss = 0.0, 0.0
        for batch_data in train_loader:
            batch_data = trainer.preprocess_batch(batch_data)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=_use_amp):
                det_loss, _ = student(batch_data)
                det_loss = det_loss.sum()

                with torch.no_grad():
                    teacher(batch_data["img"])
                student_out_feat = student_feat["feat"]
                teacher_out_feat = teacher_feat["feat"]
                # 채널 수가 다르면(프루닝됨) student 쪽 채널 수 기준으로 teacher를 1x1 conv 없이
                # adaptive하게 맞추기 위해 평균 풀링으로 채널 정렬 대신, 공간 크기만 맞추고
                # 채널은 슬라이싱으로 맞춤 (teacher 채널 >= student 채널이 보장됨).
                c = student_out_feat.shape[1]
                kd_loss = F.mse_loss(student_out_feat, teacher_out_feat[:, :c])

                loss = det_loss + args.kd_weight * kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_det_loss += det_loss.item()
            epoch_kd_loss += kd_loss.item()
        scheduler.step()
        n = len(train_loader)
        print(f"[epoch {epoch}/{args.epochs}] det_loss={epoch_det_loss/n:.4f} kd_loss={epoch_kd_loss/n:.4f}")

        save_yolo = YOLO(args.pruned_weights)
        save_yolo.model.load_state_dict(student.state_dict(), assign=True)
        atomic_save(save_yolo, os.path.join(weights_dir, "last.pt"))

        if epoch % 5 == 0 or epoch == args.epochs:
            val_yolo = YOLO(args.pruned_weights)
            val_yolo.model.load_state_dict(student.state_dict(), assign=True)
            metrics = val_yolo.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
            map50 = metrics.box.map50
            print(f"  -> validation mAP@0.5 = {map50:.4f}")

            if map50 > best_map50:
                best_map50 = map50
                no_improve_count = 0
                atomic_save(save_yolo, os.path.join(weights_dir, "best.pt"))
                print(f"  -> 새로운 best 갱신 (mAP50={best_map50:.4f}), best.pt 저장")
            else:
                no_improve_count += 5
                if no_improve_count >= args.patience:
                    print(f">>> {args.patience} epoch 동안 개선 없음, 조기 종료")
                    break

    print(f"\n프루닝+KD 완료. best mAP@0.5 = {best_map50:.4f}")
    print(f"결과: {weights_dir}")
    print("이 mAP를 순수 프루닝(finetune_pruned.py) 결과와 비교해서, capacity gap이 작을 때")
    print("KD가 실제로 도움이 되는지 확인하세요.")


if __name__ == "__main__":
    main()

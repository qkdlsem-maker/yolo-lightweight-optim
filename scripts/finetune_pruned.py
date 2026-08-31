"""
프루닝 직후(pruned.pt)의 모델을 원본 학습 데이터로 파인튜닝해서 정확도를 회복시킴.

주의(과거에 겪은 버그들, 재발 방지용 메모):
1. YOLO.val()을 호출하면 Ultralytics가 내부적으로 conv+BN을 자동 fuse(합침)해버려서,
   그 인스턴스를 재사용해 load_state_dict를 또 하면 구조 불일치(key mismatch)가 남.
   -> 저장/검증용 YOLO 인스턴스를 절대 재사용하지 않고 매번 새로 만든다.
2. state_dict를 그대로 load_state_dict()하면 프루닝으로 채널 수가 바뀐 레이어에서
   shape mismatch가 남 -> assign=True로 텐서 자체를 교체(파라미터 shape 재정의 허용).
3. 저장 도중 디스크가 가득 차거나 프로세스가 죽으면 last.pt/best.pt가 손상될 수 있음
   -> tmp 파일에 먼저 저장 후 os.replace()로 원자적 교체.

사용:
    python finetune_pruned.py \
        --pruned-weights ../outputs/yolov8n_pruned_r45/weights/pruned.pt \
        --data ../configs/bdd100k.yaml \
        --epochs 100 --batch 64 --name yolov8n_pruned_r45_finetuned
"""
import argparse
import os

import torch
from ultralytics import YOLO

import c2f_v2_utils  # noqa: F401  (pruned.pt를 unpickle하려면 C2f_v2 클래스 정의가 import되어 있어야 함)


def atomic_save(yolo_obj, path):
    tmp_path = path + ".tmp"
    try:
        yolo_obj.save(tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"  [경고] 저장 실패 ({path}): {e}. 이전 체크포인트는 그대로 유지됩니다.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned-weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20,
                         help="이 epoch 수만큼 mAP 개선이 없으면 조기 종료")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--project", type=str, default="../outputs")
    args = parser.parse_args()

    weights_dir = os.path.join(args.project, args.name, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    print(">>> 프루닝된 모델 로드 중...")
    yolo = YOLO(args.pruned_weights)
    model = yolo.model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    _use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

    # 학습 데이터로더는 Ultralytics 내부 trainer를 그대로 재사용
    from ultralytics.models.yolo.detect import DetectionTrainer
    trainer_args = dict(model=args.pruned_weights, data=args.data, epochs=1,
                         batch=args.batch, imgsz=args.imgsz, device=args.device, verbose=False)
    trainer = DetectionTrainer(overrides=trainer_args)
    trainer._setup_train(world_size=1)
    train_loader = trainer.train_loader

    # model.loss()가 내부적으로 self.args.box/.cls/.dfl 등을 참조하는데,
    # 체크포인트에서 로드한 model.args가 dict인 경우 AttributeError가 남 -> trainer의
    # SimpleNamespace 기반 args로 덮어써서 해결.
    model.args = trainer.args

    best_map50 = 0.0
    no_improve_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            batch_data = trainer.preprocess_batch(batch_data)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=_use_amp):
                loss, loss_items = model(batch_data)
                loss = loss.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"[epoch {epoch}/{args.epochs}] loss={epoch_loss/len(train_loader):.4f}")

        # 저장/검증: 매번 디스크에서 fresh하게 재로드 (fuse로 인한 구조 불일치 방지, assign=True 필수)
        save_yolo = YOLO(args.pruned_weights)
        save_yolo.model.load_state_dict(model.state_dict(), assign=True)
        atomic_save(save_yolo, os.path.join(weights_dir, "last.pt"))

        if epoch % 5 == 0 or epoch == args.epochs:
            val_yolo = YOLO(args.pruned_weights)
            val_yolo.model.load_state_dict(model.state_dict(), assign=True)
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

    print(f"\n파인튜닝 완료. best mAP@0.5 = {best_map50:.4f}")
    print(f"결과: {weights_dir}")


if __name__ == "__main__":
    main()

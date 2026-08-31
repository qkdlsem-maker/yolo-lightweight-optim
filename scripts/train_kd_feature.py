"""
Feature-based Knowledge Distillation
teacher(yolov8l, 고정) 의 중간 레이어(P3/P4/P5) 특징을 student(yolov8n)가 모방하도록 학습.

핵심 아이디어:
- YOLOv8n/l은 레이어 인덱스 구조가 동일함 (15, 18, 21번 레이어가 각각 P3/P4/P5 출력)
  -> 채널 수만 다름 (student 64/128/256ch vs teacher 256/512/512ch)
- 1x1 conv adapter로 student 채널을 teacher 채널에 맞춘 뒤 MSE loss로 특징을 모방
- 최종 loss = 기존 detection loss(box+cls+dfl) + kd_weight * feature_kd_loss

사전 검증: 레이어 인덱스/feature shape/loss-backward 파이프라인은 더미 텐서로 확인 완료.

사용:
    python train_kd_feature.py \
        --student-weights ../outputs/yolov8n_baseline/weights/best.pt \
        --teacher-weights ../outputs/yolov8l_baseline/weights/best.pt \
        --data ../configs/bdd100k.yaml \
        --epochs 100 --batch 64 --kd-weight 1.0
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset

# P3, P4, P5 특징을 뽑을 레이어 인덱스 (yolov8n/l 공통 구조)
FEAT_LAYERS = [15, 18, 21]


class FeatureAdapters(nn.Module):
    """student 채널 -> teacher 채널로 맞춰주는 1x1 conv들."""

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Conv2d(sc, tc, kernel_size=1)
            for sc, tc in zip(student_channels, teacher_channels)
        ])

    def forward(self, feats):
        return [adapter(f) for adapter, f in zip(self.adapters, feats)]


def get_channels(model, layers):
    """더미 forward 한 번 돌려서 각 레이어 출력 채널 수 확인. (모델과 같은 device 사용)"""
    feats = {}

    def mk_hook(i):
        def hook(m, inp, out):
            feats[i] = out
        return hook

    handles = [model.model[i].register_forward_hook(mk_hook(i)) for i in layers]
    model_device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(1, 3, 640, 640, device=model_device))
    for h in handles:
        h.remove()
    return [feats[i].shape[1] for i in layers]


def register_feature_hooks(model, layers, store):
    def mk_hook(i):
        def hook(m, inp, out):
            store[i] = out
        return hook
    for i in layers:
        model.model[i].register_forward_hook(mk_hook(i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-weights", type=str, required=True,
                         help="warm-start용 nano 베이스라인 가중치 (best.pt)")
    parser.add_argument("--teacher-weights", type=str, required=True,
                         help="teacher용 large 베이스라인 가중치 (best.pt)")
    parser.add_argument("--data", type=str, default="../configs/bdd100k.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kd-weight", type=float, default=1.0,
                         help="detection loss 대비 feature KD loss 가중치")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--project", type=str, default="../outputs")
    parser.add_argument("--name", type=str, default="yolov8n_kd_feature")
    parser.add_argument("--patience", type=int, default=20,
                         help="이 횟수만큼 mAP 개선 없으면 조기 종료")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # ---------- 1. teacher / student 모델 로드 ----------
    print(">>> teacher/student 로드 중...")
    teacher_yolo = YOLO(args.teacher_weights)
    student_yolo = YOLO(args.student_weights)
    # 평가/저장 전용 "깨끗한" 모델을 따로 하나 더 로드 (hook이 안 붙어있어야 torch.save가 가능함)
    # 평가/저장은 매번 fresh하게 새 YOLO 인스턴스를 만들어서 진행 (아래 참고)

    teacher: DetectionModel = teacher_yolo.model.to(device)
    student: DetectionModel = student_yolo.model.to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()

    hyp = get_cfg(DEFAULT_CFG)
    teacher.args = hyp
    student.args = hyp

    # ---------- 2. 채널 확인 후 adapter 생성 ----------
    student_channels = get_channels(student, FEAT_LAYERS)
    teacher_channels = get_channels(teacher, FEAT_LAYERS)
    print(f"student channels: {student_channels}, teacher channels: {teacher_channels}")
    adapters = FeatureAdapters(student_channels, teacher_channels).to(device)

    feats_s, feats_t = {}, {}
    register_feature_hooks(student, FEAT_LAYERS, feats_s)
    register_feature_hooks(teacher, FEAT_LAYERS, feats_t)

    # ---------- 3. 데이터로더 구성 ----------
    print(">>> 데이터셋 로드 중...")
    data_dict = check_det_dataset(args.data)
    train_set = build_yolo_dataset(hyp, data_dict["train"], args.batch, data_dict, mode="train", rect=False)
    train_loader = build_dataloader(train_set, args.batch, args.workers, shuffle=True)

    # ---------- 4. optimizer (student + adapter 파라미터 모두 학습) ----------
    params = list(student.parameters()) + list(adapters.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.937, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    best_map50 = 0.0
    no_improve_count = 0
    csv_path = os.path.join(save_dir, "kd_train_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,det_loss,kd_loss,total_loss,epoch_time_sec\n")

    print(">>> KD 학습 시작...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_det, running_kd, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
            batch["cls"] = batch["cls"].to(device)
            batch["bboxes"] = batch["bboxes"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds = student(batch["img"])
                with torch.no_grad():
                    _ = teacher(batch["img"])

                det_loss, _ = student.loss(batch, preds)
                det_loss = det_loss.sum()

                adapted_feats = adapters([feats_s[i] for i in FEAT_LAYERS])
                kd_loss = sum(
                    nn.functional.mse_loss(sf, feats_t[i].detach())
                    for sf, i in zip(adapted_feats, FEAT_LAYERS)
                )

                total_loss = det_loss + args.kd_weight * kd_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_det += det_loss.item()
            running_kd += kd_loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"  epoch {epoch} batch {n_batches}/{len(train_loader)} "
                      f"det_loss={running_det/n_batches:.4f} kd_loss={running_kd/n_batches:.4f}")

        epoch_time = time.time() - epoch_start
        avg_det = running_det / max(n_batches, 1)
        avg_kd = running_kd / max(n_batches, 1)
        print(f"[Epoch {epoch}/{args.epochs}] det_loss={avg_det:.4f} kd_loss={avg_kd:.4f} "
              f"time={epoch_time:.1f}s")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_det:.4f},{avg_kd:.4f},{avg_det + args.kd_weight*avg_kd:.4f},{epoch_time:.1f}\n")

        # 체크포인트 저장 (매 epoch last, 검증 갱신 시 best)
        # 주의: student에는 KD용 forward hook이 붙어있어서 그대로 pickle(save)하면 에러 발생.
        # -> hook이 없는 "깨끗한" 모델을 매번 새로 만들어서 저장/평가에 사용.
        #    (주의: YOLO.val()이 내부적으로 conv+BN을 fuse해서 구조를 영구 변경하므로,
        #     인스턴스를 재사용하면 다음 load_state_dict에서 키 불일치가 남 -> 매번 새로 로드)
        eval_yolo = YOLO(args.student_weights)
        eval_yolo.model.load_state_dict(student.state_dict(), assign=True)
        torch.save(student.state_dict(), os.path.join(weights_dir, "last_student_state_dict.pt"))
        eval_yolo.save(os.path.join(weights_dir, "last.pt"))

        # 5 epoch마다 검증
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = eval_yolo.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
            map50 = metrics.box.map50
            print(f"  -> validation mAP@0.5 = {map50:.4f}")

            if map50 > best_map50:
                best_map50 = map50
                no_improve_count = 0
                torch.save(student.state_dict(), os.path.join(weights_dir, "best_student_state_dict.pt"))
                eval_yolo.save(os.path.join(weights_dir, "best.pt"))
                print(f"  -> 새로운 best 갱신 (mAP50={best_map50:.4f}), best.pt 저장")
            else:
                no_improve_count += 1

            if no_improve_count * 5 >= args.patience:
                print(f"조기 종료: {args.patience} epoch 동안 mAP 개선 없음.")
                break

    print(f"\nKD 학습 완료. best mAP@0.5 = {best_map50:.4f}")
    print(f"최종 가중치: {weights_dir}/best.pt")


if __name__ == "__main__":
    main()

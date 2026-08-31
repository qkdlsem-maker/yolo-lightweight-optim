"""
prune_model.py로 프루닝한 모델을 파인튜닝(재학습)해서 떨어진 mAP를 회복.

KD 스크립트(train_kd_feature.py)에서 검증했던 안전 패턴을 그대로 재사용:
- 매 검증/저장마다 디스크에서 fresh하게 새 YOLO 인스턴스를 만듦
  (YOLO.val()이 내부적으로 conv+BN을 fuse해서 구조를 영구 변경 -> 인스턴스 재사용 시 다음
   load_state_dict에서 키 불일치 발생하는 버그를 피하기 위함)
- load_state_dict(..., assign=True)로 inference tensor in-place 에러 방지

사용:
    python finetune_pruned.py \
        --pruned-weights ../outputs/yolov8n_pruned_r30/weights/pruned.pt \
        --data ../configs/bdd100k.yaml \
        --epochs 100 --batch 64
"""
import argparse
import os
import time

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset

# pruned 모델 안의 C2f_v2를 pickle이 다시 풀 수 있도록 반드시 import 필요
import c2f_v2_utils  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned-weights", type=str, required=True,
                         help="prune_model.py가 만든 (파인튜닝 전) pruned.pt")
    parser.add_argument("--data", type=str, default="../configs/bdd100k.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--project", type=str, default="../outputs")
    parser.add_argument("--name", type=str, default="yolov8n_pruned_finetuned")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    print(">>> pruned 모델 로드 중...")
    model_yolo = YOLO(args.pruned_weights)
    model = model_yolo.model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    hyp = get_cfg(DEFAULT_CFG)
    model.args = hyp

    print(">>> 데이터셋 로드 중...")
    data_dict = check_det_dataset(args.data)
    train_set = build_yolo_dataset(hyp, data_dict["train"], args.batch, data_dict, mode="train", rect=False)
    train_loader = build_dataloader(train_set, args.batch, args.workers, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.937, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    best_map50 = 0.0
    no_improve_count = 0
    csv_path = os.path.join(save_dir, "finetune_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,det_loss,epoch_time_sec\n")

    print(">>> 파인튜닝 시작...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss, n_batches = 0.0, 0

        for batch in train_loader:
            batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
            batch["cls"] = batch["cls"].to(device)
            batch["bboxes"] = batch["bboxes"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(batch["img"])
                loss, _ = model.loss(batch, preds)
                loss = loss.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  epoch {epoch} batch {n_batches}/{len(train_loader)} "
                      f"loss={running_loss/n_batches:.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(n_batches, 1)
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.4f} time={epoch_time:.1f}s")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{epoch_time:.1f}\n")

        # 저장/검증: 매번 디스크에서 fresh하게 재로드 (fuse로 인한 구조 불일치 방지)
        eval_yolo = YOLO(args.pruned_weights)
        eval_yolo.model.load_state_dict(model.state_dict(), assign=True)
        eval_yolo.save(os.path.join(weights_dir, "last.pt"))

        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = eval_yolo.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
            map50 = metrics.box.map50
            print(f"  -> validation mAP@0.5 = {map50:.4f}")

            if map50 > best_map50:
                best_map50 = map50
                no_improve_count = 0
                eval_yolo.save(os.path.join(weights_dir, "best.pt"))
                print(f"  -> 새로운 best 갱신 (mAP50={best_map50:.4f}), best.pt 저장")
            else:
                no_improve_count += 1

            if no_improve_count * 5 >= args.patience:
                print(f"조기 종료: {args.patience} epoch 동안 mAP 개선 없음.")
                break

    print(f"\n파인튜닝 완료. best mAP@0.5 = {best_map50:.4f}")
    print(f"최종 가중치: {weights_dir}/best.pt")


if __name__ == "__main__":
    main()
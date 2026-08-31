"""
학습된 모델의 mAP / FPS / 파라미터 수 / 모델 용량을 측정.
논문의 '정확도-속도-경량화 트레이드오프' 표에 바로 쓸 수 있는 값들을 출력.

사용:
    python eval_baseline.py --weights ../outputs/yolov8n_baseline/weights/best.pt
"""
import argparse
import time
import os
import torch
from ultralytics import YOLO


def measure_fps(model, imgsz=640, n_iters=200, device=0):
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    model.model.to(device).eval()

    # 워밍업
    with torch.no_grad():
        for _ in range(20):
            model.model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            model.model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    fps = n_iters / elapsed
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="../configs/bdd100k.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(args.weights)

    # 1. mAP 측정 (val set 기준)
    print(">>> mAP 측정 중...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    # 2. FPS 측정
    print(">>> FPS 측정 중...")
    fps = measure_fps(model, imgsz=args.imgsz, device=args.device)

    # 3. 파라미터 수 / 모델 용량
    n_params = sum(p.numel() for p in model.model.parameters())
    file_size_mb = os.path.getsize(args.weights) / (1024 * 1024)

    print("\n========== 결과 요약 ==========")
    print(f"가중치 파일        : {args.weights}")
    print(f"mAP@0.5            : {map50:.4f}")
    print(f"mAP@0.5:0.95        : {map50_95:.4f}")
    print(f"FPS (배치=1, {args.imgsz}px) : {fps:.2f}")
    print(f"파라미터 수         : {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"모델 용량           : {file_size_mb:.2f} MB")
    print("================================")

    # 논문용 CSV 누적 기록
    csv_path = "../outputs/results_summary.csv"
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if header_needed:
            f.write("weights,mAP50,mAP50_95,fps,params_M,size_MB\n")
        f.write(f"{args.weights},{map50:.4f},{map50_95:.4f},{fps:.2f},{n_params/1e6:.2f},{file_size_mb:.2f}\n")
    print(f"\n결과가 {csv_path} 에 누적 저장되었습니다.")


if __name__ == "__main__":
    main()

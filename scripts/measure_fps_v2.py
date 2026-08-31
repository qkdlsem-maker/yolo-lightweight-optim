"""
기존 eval_baseline.py의 measure_fps()가 Python time.time() + 반복 200회로 재다 보니
런마다 258 vs 494, 343 vs 452처럼 같은 모델인데도 값이 크게 흔들리는 문제가 있었음.

원인 후보: (1) cudnn.benchmark 미설정으로 매 런마다 커널 선택이 달라짐,
(2) 워밍업(20회)이 GPU 클럭이 boost 상태로 안정화되기엔 부족함,
(3) Python 레벨 time.time()은 커널 큐잉/디스패치 오버헤드까지 섞여서 잡음.

개선: torch.cuda.Event로 GPU 실행시간만 정확히 측정 + 워밍업 100회 + 본측정 1000회 +
      cudnn.benchmark=True + 3회 반복 측정 후 median 사용 (한 번의 튀는 값에 안 흔들리게).

사용 (기존 eval_baseline.py에서 measure_fps 함수만 이 파일 것으로 바꿔치기,
또는 이 파일을 그냥 단독으로 실행해도 됨):
    python measure_fps_v2.py --weights ../outputs/yolov8n_pruned_r45_finetuned/weights/best.pt
    python measure_fps_v2.py --weights ../outputs/yolov8n_pruned_r45_kd/weights/best.pt
"""
import argparse
import statistics

import torch
from ultralytics import YOLO


def measure_fps(model, imgsz=640, warmup=100, n_iters=1000, n_trials=10, device=0):
    torch.backends.cudnn.benchmark = True
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    m = model.model.to(device).eval()

    # 체크포인트마다 저장 시점에 conv+BN이 fuse된 상태/안 된 상태가 섞여있으면
    # (fuse된 쪽이 항상 더 빠름) 같은 구조인데도 재현 가능하게 FPS가 벌어짐.
    # 모든 모델을 여기서 강제로 동일하게 fuse시켜서 이 편차를 원천 제거.
    if hasattr(m, "fuse"):
        m.fuse()

    with torch.no_grad():
        for _ in range(warmup):
            m(dummy)
    torch.cuda.synchronize()

    trial_fps = []
    for trial in range(n_trials):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_evt.record()
        with torch.no_grad():
            for _ in range(n_iters):
                m(dummy)
        end_evt.record()
        torch.cuda.synchronize()

        elapsed_ms = start_evt.elapsed_time(end_evt)  # GPU 실행시간(ms), 커널 큐잉 오버헤드 최소화
        fps = n_iters / (elapsed_ms / 1000)
        trial_fps.append(fps)
        print(f"  trial {trial+1}/{n_trials}: {fps:.2f} FPS")

    return statistics.median(trial_fps), statistics.stdev(trial_fps) if len(trial_fps) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(args.weights)
    print(">>> FPS 측정 중 (워밍업 100회 + 1000회x3 trial, CUDA event 기반)...")
    median_fps, std_fps = measure_fps(model, imgsz=args.imgsz, device=args.device)

    print("\n========== FPS 결과 ==========")
    print(f"가중치 파일 : {args.weights}")
    print(f"FPS (median of 3 trials) : {median_fps:.2f} (std={std_fps:.2f})")
    print("================================")


if __name__ == "__main__":
    main()

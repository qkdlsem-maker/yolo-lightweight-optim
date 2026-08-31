"""
FP16 PTQ를 ONNX/TensorRT export 없이 Ultralytics 네이티브 half=True로 측정.
(onnxruntime-gpu가 서버 CUDA 버전과 안 맞아 export 경로가 계속 실패해서,
 더 단순하고 의존성 적은 이 경로로 대체함.)

INT8은 이 스크립트에서 다루지 않음 -- TensorRT가 이 서버의 CUDA 12.1 툴체인과
버전이 맞물려야 하는데 (onnxruntime-gpu 때와 동일한 문제 재발 가능성 높음),
제대로 맞추려면 시간이 많이 들어서 이번 실험 범위에서는 FP16까지만 하고
INT8은 "TensorRT 버전 미스매치로 시도했으나 보류, 향후 연구 과제"로 논문에
명시하는 쪽을 권장.

사용:
    python quantize_fp16_native.py \
        --weights ../outputs/yolov8n_pruned_r45_finetuned/weights/best.pt \
        --data ../configs/bdd100k.yaml
"""
import argparse
import statistics

import torch
from ultralytics import YOLO

import c2f_v2_utils  # noqa: F401


def measure_fps_half(model, imgsz=640, warmup=100, n_iters=1000, n_trials=5, device=0):
    torch.backends.cudnn.benchmark = True
    m = model.model.to(device).half().eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=torch.half)

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
        elapsed_ms = start_evt.elapsed_time(end_evt)
        fps = n_iters / (elapsed_ms / 1000)
        trial_fps.append(fps)
        print(f"  trial {trial+1}/{n_trials}: {fps:.2f} FPS")
    return statistics.mean(trial_fps), (statistics.stdev(trial_fps) if len(trial_fps) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    print(">>> FP32 (원본) mAP 측정 중...")
    model_fp32 = YOLO(args.weights)
    metrics_fp32 = model_fp32.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
    map50_fp32 = metrics_fp32.box.map50

    print("\n>>> FP16 mAP 측정 중 (Ultralytics 네이티브 half=True)...")
    model_fp16 = YOLO(args.weights)
    metrics_fp16 = model_fp16.val(data=args.data, imgsz=args.imgsz, device=args.device, half=True, verbose=False)
    map50_fp16 = metrics_fp16.box.map50

    print("\n>>> FP32 FPS 측정 중...")
    fp32_model_for_fps = YOLO(args.weights)
    m32 = fp32_model_for_fps.model.to(args.device).eval()
    dummy32 = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.no_grad():
        for _ in range(100):
            m32(dummy32)
    torch.cuda.synchronize()
    trial_fps32 = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        with torch.no_grad():
            for _ in range(1000):
                m32(dummy32)
        e.record(); torch.cuda.synchronize()
        trial_fps32.append(1000 / (s.elapsed_time(e) / 1000))
    fps32_mean, fps32_std = statistics.mean(trial_fps32), statistics.stdev(trial_fps32)

    print("\n>>> FP16 FPS 측정 중...")
    fp16_model_for_fps = YOLO(args.weights)
    fps16_mean, fps16_std = measure_fps_half(fp16_model_for_fps, imgsz=args.imgsz, device=args.device)

    print("\n========== FP16 PTQ 결과 요약 ==========")
    print(f"FP32: mAP@0.5={map50_fp32:.4f}  FPS={fps32_mean:.1f}±{fps32_std:.1f}")
    print(f"FP16: mAP@0.5={map50_fp16:.4f}  FPS={fps16_mean:.1f}±{fps16_std:.1f}")
    print("=========================================")
    print("\nINT8은 이번엔 스킵 (TensorRT 버전 미스매치 이슈). 필요하면 별도로 처리 논의.")


if __name__ == "__main__":
    main()

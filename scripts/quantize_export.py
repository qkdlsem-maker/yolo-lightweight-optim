"""
Quantization (PTQ) 실험: FP16, INT8.

프루닝/KD와는 다른 축의 경량화 기법이라 SCI 컨트리뷰션 보강용으로 추가.
가장 성능 좋은 후보(프루닝 0.45 파인튜닝 모델 등)에 적용해서
mAP 하락 대비 추가 speedup이 있는지 확인하는 게 목적.

Ultralytics의 model.export()가 TensorRT engine으로 내보내면서 FP16/INT8을
자동으로 처리해줌 (TensorRT가 서버에 설치되어 있어야 함).

주의:
- INT8은 대표 이미지로 calibration이 필요해서 --data(yaml)를 그대로 넘기면
  Ultralytics가 val 데이터 일부를 자동으로 calibration에 사용함.
- TensorRT가 없으면 이 스크립트의 int8/engine 경로는 실패함 -> 그 경우
  ONNX + half(FP16)만이라도 시도하도록 되어 있음 (아래 --skip-int8 참고).

사용:
    # FP16 + INT8 둘 다
    python quantize_export.py --weights ../outputs/yolov8n_pruned_r45_finetuned/weights/best.pt \
        --data ../configs/bdd100k.yaml

    # TensorRT/INT8 없이 FP16만
    python quantize_export.py --weights ../outputs/yolov8n_pruned_r45_finetuned/weights/best.pt \
        --data ../configs/bdd100k.yaml --skip-int8
"""
import argparse
import os
import statistics

import torch
from ultralytics import YOLO

import c2f_v2_utils  # noqa: F401  (pruned 계열 가중치 unpickle에 필요)


def measure_fps_engine(model, imgsz=640, warmup=100, n_iters=1000, n_trials=5):
    """CUDA event 기반 FPS 측정 (measure_fps_v2.py와 동일 방법론, engine/onnx 모델에도 재사용)."""
    dummy = torch.randn(1, 3, imgsz, imgsz)
    if torch.cuda.is_available():
        dummy = dummy.cuda()

    with torch.no_grad():
        for _ in range(warmup):
            model.predict(dummy, verbose=False)
    torch.cuda.synchronize()

    trial_fps = []
    for trial in range(n_trials):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        with torch.no_grad():
            for _ in range(n_iters):
                model.predict(dummy, verbose=False)
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
    parser.add_argument("--skip-int8", action="store_true",
                         help="TensorRT가 없거나 INT8 calibration을 건너뛰고 싶을 때")
    args = parser.parse_args()

    base = YOLO(args.weights)
    orig_params = sum(p.numel() for p in base.model.parameters())
    print(f">>> 원본 모델 파라미터 수: {orig_params:,}")

    results = {}

    # ---------- FP16 (ONNX) ----------
    print("\n>>> FP16 ONNX export 중...")
    try:
        fp16_path = base.export(format="onnx", half=True, imgsz=args.imgsz, device=0)
        fp16_model = YOLO(fp16_path)
        print(">>> FP16 모델 mAP 측정 중...")
        metrics = fp16_model.val(data=args.data, imgsz=args.imgsz, device=0, verbose=False)
        map50 = metrics.box.map50
        print(">>> FP16 모델 FPS 측정 중...")
        fps_mean, fps_std = measure_fps_engine(fp16_model, imgsz=args.imgsz)
        results["FP16"] = dict(mAP50=map50, fps_mean=fps_mean, fps_std=fps_std, path=fp16_path)
    except Exception as e:
        print(f"  [FP16 실패] {e}")
        results["FP16"] = None

    # ---------- INT8 (TensorRT engine) ----------
    if not args.skip_int8:
        print("\n>>> INT8 TensorRT engine export 중 (calibration에 val 데이터 사용, 시간 좀 걸림)...")
        try:
            int8_path = base.export(format="engine", int8=True, data=args.data,
                                     imgsz=args.imgsz, device=0)
            int8_model = YOLO(int8_path)
            print(">>> INT8 모델 mAP 측정 중...")
            metrics = int8_model.val(data=args.data, imgsz=args.imgsz, device=0, verbose=False)
            map50 = metrics.box.map50
            print(">>> INT8 모델 FPS 측정 중...")
            fps_mean, fps_std = measure_fps_engine(int8_model, imgsz=args.imgsz)
            results["INT8"] = dict(mAP50=map50, fps_mean=fps_mean, fps_std=fps_std, path=int8_path)
        except Exception as e:
            print(f"  [INT8 실패] {e}")
            print("  TensorRT 미설치가 원인일 가능성이 높음. 'pip install tensorrt' 후 재시도하거나")
            print("  --skip-int8 옵션으로 FP16 결과만 우선 사용하세요.")
            results["INT8"] = None

    print("\n========== Quantization 결과 요약 ==========")
    print(f"원본 (FP32) 파라미터: {orig_params:,}")
    for name, r in results.items():
        if r is None:
            print(f"{name}: 실패")
        else:
            print(f"{name}: mAP@0.5={r['mAP50']:.4f}  FPS={r['fps_mean']:.1f}±{r['fps_std']:.1f}  ({r['path']})")
    print("=============================================")


if __name__ == "__main__":
    main()

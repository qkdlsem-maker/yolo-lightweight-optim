"""
YOLOv8 베이스라인 학습 (RTX 4090 x2 기준)

사용 예:
    python train_baseline.py --model yolov8n.pt --gpus 0,1 --epochs 100
    python train_baseline.py --model yolov8l.pt --gpus 0,1 --epochs 100

n(nano) / l(large) 둘 다 학습해두면, 추후 지식증류 실험에서
l -> teacher, n -> student 로 바로 활용 가능.
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt")
    parser.add_argument("--data", type=str, default="../configs/bdd100k.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32,
                         help="4090 24GB x2 기준 n모델은 64~128, l모델은 32 권장")
    parser.add_argument("--gpus", type=str, default="0,1", help="예: '0' 또는 '0,1'")
    parser.add_argument("--project", type=str, default="../outputs")
    args = parser.parse_args()

    run_name = f"{args.model.replace('.pt', '')}_baseline"

    model = YOLO(args.model)
    device = [int(g) for g in args.gpus.split(",")] if "," in args.gpus else args.gpus

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=run_name,
        patience=20,        # early stopping
        workers=8,
        cache=False,        # 데이터셋이 크면 디스크 캐시 대신 False 권장 (RAM 부족 방지)
        exist_ok=True,
    )

    print(f"\n학습 완료. 가중치 경로: {args.project}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()

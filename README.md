# 자율주행 경량 객체 탐지 최적화 실험

KCI 논문: "지식증류/경량화 기반 자율주행 실시간 객체 탐지 최적화"

## 실험 환경
- Linux, RTX 4090 x2
- CUDA 12.x, PyTorch 2.x, ultralytics(YOLOv8)

## 폴더 구조
```
yolo_lightweight_optim/
├── configs/
│   └── bdd100k.yaml          # 데이터셋/클래스 정의
├── scripts/
│   ├── prepare_bdd100k.py    # BDD100K 라벨 -> YOLO 포맷 변환
│   ├── train_baseline.py     # 베이스라인 YOLOv8 학습
│   └── eval_baseline.py      # mAP / FPS / 파라미터 수 측정
├── data/
│   └── DOWNLOAD_INSTRUCTIONS.md
├── outputs/                  # 학습 결과, 가중치 저장 위치
├── logs/                     # 학습 로그
├── requirements.txt
└── setup_env.sh
```

## 실행 순서

### 1. 서버로 파일 옮기기
로컬(제 작업 환경)에서 받은 zip을 서버로 전송:
```bash
# 사용자 PC에서 (mobaxterm 로컬 터미널 또는 scp 지원 툴 사용)
scp yolo_lightweight_optim.zip [email protected]:~/choihyerim1129/
```
서버 접속 후:
```bash
cd ~/choihyerim1129
unzip yolo_lightweight_optim.zip
cd yolo_lightweight_optim
```

### 2. 환경 세팅
```bash
bash setup_env.sh
conda activate yolo-optim
```

### 3. 데이터셋 다운로드
`data/DOWNLOAD_INSTRUCTIONS.md` 참고 (BDD100K는 회원가입 후 수동 다운로드 필요)

### 4. 라벨 변환 (BDD100K JSON -> YOLO txt)
```bash
python scripts/prepare_bdd100k.py \
    --bdd-root /path/to/bdd100k \
    --out-root ./data/bdd100k_yolo
```

### 5. 베이스라인 학습 (2-GPU)
```bash
python scripts/train_baseline.py --model yolov8n.pt --gpus 0,1 --epochs 100
python scripts/train_baseline.py --model yolov8l.pt --gpus 0,1 --epochs 100
```
n(nano)과 l(large) 두 개를 베이스라인으로 잡아두면, 추후 경량화 기법(pruning/distillation) 적용 전후 비교가 명확해집니다 (l을 teacher, n을 student로 지식증류 실험 가능).

### 6. 평가 (mAP, FPS, 파라미터 수)
```bash
python scripts/eval_baseline.py --weights outputs/yolov8n_baseline/weights/best.pt
python scripts/eval_baseline.py --weights outputs/yolov8l_baseline/weights/best.pt
```

## 다음 단계 (베이스라인 완료 후)
- Knowledge Distillation: yolov8l(teacher) -> yolov8n(student)
- Pruning: torch-pruning 라이브러리로 채널 프루닝
- Quantization: INT8 양자화 후 TensorRT 변환 (Jetson급 엣지 배포 가정 시 설득력 ↑)
- 위 세 기법을 각각 적용 후 mAP-FPS-모델크기 트레이드오프 표로 정리 → 논문 실험 섹션의 핵심 그림

# 자율주행 경량 객체 탐지 최적화 실험

IEEE Access 투고를 준비하는 YOLOv8 경량화 연구입니다. 게재 또는 심사 완료를 의미하지 않습니다.


## 새 보완 단계: 검증 라벨 재학습과 외부 평가 (진행 중)

미확인 학습 137장을 제외한 69,863장으로 학생·교사를 COCO 초기값부터 새로 학습하기 시작했습니다. 이후 대조군/MSE/CWD 9회 비교와 KITTI 외부 평가를 진행합니다. 기존 완료 결과는 별도로 보존하며, 이 새 단계의 결과는 아직 완료되지 않았습니다. [사전 명시 조건](experiments/clean_followup/protocol.md)과 [검증된 사전 점검 기록·현재 범위](artifacts/clean_followup_20260929/STATUS.md)를 공개합니다. KITTI 공개 학습 파티션을 본 연구의 외부 시험에 사용하는 것이며 공식 비공개 시험 점수라고 주장하지 않습니다.

## 2026-09 검증 및 재현 자료

중요: 기존 standalone KD 두 실행은 학생 가중치를 업데이트하지 않았습니다. 해당 점수를 KD 효과 또는 교사 크기의 효과로 해석하면 안 됩니다. 수정 코드에는 학생 학습 상태 복구, BN 통계를 바꾸지 않는 채널 확인, 검증용 가중치 분리, 명시적 데이터 시드가 포함됩니다.

- [완료된 재평가·감사·실험 기록](artifacts/ieee_access_20260929/README.md): 기존 모델 9개 평가, 반복 실행시간, 원시 수치와 코드.
- 9월 22일 추가 학습은 6회 완료됐지만 데이터로더 내부 시드가 고정되었습니다. 대조군 3회는 동일한 상태입니다. 독립 3시드 결과가 아니라 고정 데이터 조건에서 KD 어댑터 초기화를 바꾼 결과로만 해석합니다.
- [독립 데이터 시드와 CWD 비교의 사전 명시 프로토콜](experiments/protocol_20260929.md): 대조군·MSE·CWD × 독립 데이터 seed 0/1/2의 총 9회 20 epoch 학습과 최종 검증을 완료했습니다. 동일 시드 입력 일치, 시드 간 차이, 실제 학생 갱신과 최종 수치 상태를 검증했습니다.
- [후속 실험 실행 코드](experiments/confirmatory_kd.py): 대조군 / MSE / CWD, 명시적인 sampler·worker 시드, 입력 해시 비교, 가중치 업데이트 검사, 고정 최종 EMA 평가 및 재시작 지원.
- [데이터 출처 감사](artifacts/ieee_access_20260929/DATA_PROVENANCE.md): 2018년 원본 형식 JSON과 전수 대조하여 검증 185,526개 정상 상자 및 학습 69,863장 라벨의 일치를 확인했습니다. 면적 0 상자와 미확인 학습 137장을 구분했고, 재구성 검증셋에서 기존 7개와 새 최종 모델 9개, 총 16개 모델의 라벨 민감도와 주야간 분석을 완료했습니다. 원본 데이터는 유지했습니다. 출처 불명의 test 라벨은 사용하지 않습니다.

검증 환경: Python 3.12.3, PyTorch 2.5.1+cu121, Ultralytics 8.2.103, Torch-Pruning 1.6.0, RTX 4090. 기존 결과의 원본 코드 기준은 커밋 `4f5be41e22ea8696800e94611688777a7cbb883a`입니다. 수정 코드가 과거의 결과를 생성했다고 소급해서 주장하지 않습니다. 데이터와 체크포인트 확보가 별도로 필요하며 결과 파일의 SHA-256으로 모델을 식별할 수 있습니다.

```bash
# 의도적으로 재현할 때만 새 출력 경로를 지정하는 예
python experiments/confirmatory_kd.py --arm control --seed 0 --device 0 --out-root replication/confirmatory
python experiments/confirmatory_kd.py --arm mse --seed 0 --device 0 --out-root replication/confirmatory
python experiments/confirmatory_kd.py --arm cwd --seed 0 --device 0 --out-root replication/confirmatory
```

완료된 주평가 AP50:95 평균은 대조군 0.259121, MSE 0.258870, CWD 0.258893입니다. 대조군 대비 평균 차이는 MSE -0.0251 pp, CWD -0.0229 pp로 일관된 개선을 확인하지 못했습니다. [모든 시드의 결과](artifacts/ieee_access_20260929/followup/confirmatory_results.csv)와 [수치 상태 검사](artifacts/ieee_access_20260929/followup/numerical_state_audit.json)를 공개합니다. 데이터 시드는 독립이지만 시작 기준 모델은 하나이며, validation 재사용·학습 137장 라벨 미검증·2018 라벨의 한계를 유지합니다. 위 명령은 재현을 위한 예시이며 이미 완료된 실행을 덮어쓰지 마십시오.

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

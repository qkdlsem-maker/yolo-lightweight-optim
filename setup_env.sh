#!/bin/bash
# 자율주행 최적화 실험 환경 세팅
# 사용: bash setup_env.sh

set -e

ENV_NAME="yolo-optim"

echo ">>> conda 환경 생성: $ENV_NAME (python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

# conda activate가 스크립트 내에서 동작하도록
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ">>> PyTorch (CUDA 12.1 기준) 설치"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ">>> 나머지 패키지 설치"
pip install -r requirements.txt

echo ">>> GPU 인식 확인"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

echo ">>> 완료. 'conda activate $ENV_NAME' 으로 활성화하세요."

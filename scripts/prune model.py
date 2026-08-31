"""
YOLOv8n 베이스라인을 구조적으로 채널 프루닝(channel pruning).

핵심:
- C2f 블록을 pruning 호환 버전(C2f_v2)으로 먼저 교체 (c2f_v2_utils.py 참고)
- Detect(출력 head)는 프루닝 대상에서 제외 (클래스 수/구조가 깨지면 안 되므로)
- GroupNormPruner + GroupMagnitudeImportance(L2 norm 기준)로 채널 그룹 단위 프루닝

사전 검증: 더미 모델로 params 3.01M -> 1.69M(56%), MACs 60.9%까지 줄어드는 것과
저장/재로드/backward가 정상 동작하는 것 확인 완료.

사용:
    python prune_model.py \
        --weights ../outputs/yolov8n_baseline/weights/best.pt \
        --pruning-ratio 0.3 \
        --output ../outputs/yolov8n_pruned_r30/weights/pruned.pt
"""
import argparse
import os

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import initialize_weights
import torch_pruning as tp

from c2f_v2_utils import replace_c2f_with_c2f_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True,
                         help="프루닝할 베이스라인 가중치 (예: yolov8n_baseline/weights/best.pt)")
    parser.add_argument("--pruning-ratio", type=float, default=0.3,
                         help="채널을 얼마나 줄일지 (0.3 = 채널의 30% 제거)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(">>> 모델 로드 중...")
    yolo = YOLO(args.weights)
    model = yolo.model
    model.train()

    print(">>> C2f -> C2f_v2 (pruning 호환 버전) 교체 중...")
    replace_c2f_with_c2f_v2(model)
    initialize_weights(model)
    for p in model.parameters():
        p.requires_grad = True

    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz)

    # 교체 직후 정상 동작(수학적으로 원본과 동일해야 함) 확인
    model.eval()
    with torch.no_grad():
        model(example_inputs)
    model.train()

    ignored_layers = [m for m in model.modules() if isinstance(m, Detect)]

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    print(f">>> 프루닝 진행 중 (목표 비율: {args.pruning_ratio})...")
    pruner = tp.pruner.GroupNormPruner(
        model,
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(),
        iterative_steps=1,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()

    new_macs, new_params = tp.utils.count_ops_and_params(model, example_inputs)

    print("\n========== 프루닝 결과 ==========")
    print(f"파라미터: {base_params:,} -> {new_params:,} ({100*new_params/base_params:.1f}%)")
    print(f"MACs    : {base_macs:,.0f} -> {new_macs:,.0f} ({100*new_macs/base_macs:.1f}%)")
    print("=================================\n")

    # 프루닝 직후 forward 재확인 (구조가 깨지지 않았는지)
    model.eval()
    with torch.no_grad():
        model(example_inputs)
    print(">>> 프루닝 후 forward 정상 확인")

    yolo.model = model
    yolo.save(args.output)
    print(f"\n프루닝된 (파인튜닝 전) 모델 저장 완료: {args.output}")
    print("이 상태는 아직 파인튜닝 전이라 mAP가 많이 떨어져 있는 게 정상입니다.")
    print("다음 단계: finetune_pruned.py 로 파인튜닝 진행하세요.")


if __name__ == "__main__":
    main()
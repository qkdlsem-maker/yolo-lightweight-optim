"""
YOLOv8의 C2f 블록은 forward에서 self.cv1(x).chunk(2,1)로 텐서를 반으로 쪼개는데,
이 동적 split 연산 때문에 torch-pruning이 채널 의존성을 제대로 추적하지 못함
(VainF/Torch-Pruning 공식 이슈 #147에서 다뤄진 문제).

해결책: cv1 하나로 처리하던 것을 cv0/cv1 두 개의 별도 conv로 나눠서
동일한 결과를 내지만 pruning 라이브러리가 추적 가능한 형태(C2f_v2)로 교체.
가중치는 기존 cv1의 절반씩을 그대로 나눠 옮겨서 수학적으로 100% 동일하게 동작함.

주의: 이 파일에서 정의한 C2f_v2 클래스는 pruning 이후 모델을 저장(torch.save)하면
그 안에 그대로 pickle되므로, 이후 이 pruned 모델을 불러오는 모든 스크립트
(finetune_pruned.py, distill_pruned.py, eval_baseline.py 등)를 이 파일과 같은
디렉토리(scripts/)에서 실행해야 함 (그래야 pickle이 이 모듈을 자동으로 다시 import 할 수 있음).
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck, C2f


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


class C2f_v2(nn.Module):
    """C2f와 수학적으로 동일하지만, chunk() 대신 별도 conv 2개(cv0/cv1)로 split을 대체."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    old_weight = state_dict["cv1.conv.weight"]
    half = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half]
    state_dict_v2["cv1.conv.weight"] = old_weight[half:]

    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half:]

    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    """모델 안의 모든 C2f를 C2f_v2로 재귀적으로 교체 (가중치는 그대로 이전)."""
    for name, child in module.named_children():
        if isinstance(child, C2f):
            shortcut = infer_shortcut(child.m[0])
            c2f_v2 = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=shortcut,
                g=child.m[0].cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels,
            )
            transfer_weights(child, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child)


def get_channels(module):
    """모듈이 위치한 device를 CPU로 단정하지 말고 파라미터에서 직접 추론
    (이전에 CPU로 하드코딩했다가 GPU 학습 중 device mismatch 에러 발생했던 버그 수정판)."""
    param = next(module.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    return device

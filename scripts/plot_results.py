"""
results_summary.csv를 읽어서 논문용 트레이드오프 그래프를 그림.
- 왼쪽: 파라미터 수 vs mAP@0.5 (프루닝 곡선 + KD/baseline/teacher 비교점)
- 오른쪽: FPS vs mAP@0.5 (속도-정확도 트레이드오프)

각 실험을 어떤 그룹(baseline/teacher/KD/pruning)으로 표시할지는
아래 EXPERIMENT_INFO에서 weights 경로의 일부 문자열로 매칭해서 결정.
새 실험을 추가했으면 이 딕셔너리에 한 줄만 추가하면 됨.

사용:
    python plot_results.py --csv ../outputs/results_summary.csv --output ../outputs/figures/tradeoff.png
"""
import argparse
import os
import csv as csv_module

import matplotlib.pyplot as plt

# key: weights 경로에 포함된 문자열 (부분 일치로 찾음)
# value: (그룹, 표시 이름, 라벨 텍스트 offset(dx, dy) - 겹침 방지용으로 점마다 다르게 줌)
EXPERIMENT_INFO = {
    "yolov8l_baseline":        ("teacher", "YOLOv8l (Teacher)", (8, -14)),
    "yolov8n_baseline":        ("baseline", "YOLOv8n (Baseline)", (8, 10)),
    "yolov8n_kd_feature":      ("kd", "KD (weight=1.0)", (8, -14)),
    "yolov8n_kd_w03":          ("kd", "KD (weight=0.3)", (8, -26)),
    "pruned_r15_finetuned":    ("pruning", "Pruning (ratio=0.15)", (7, 5)),
    "pruned_finetuned":        ("pruning", "Pruning (ratio=0.30)", (7, 5)),  # 기본 이름(r30)
    "pruned_r45_finetuned":    ("pruning", "Pruning (ratio=0.45)", (7, 5)),
}

GROUP_STYLE = {
    "baseline": dict(color="black", marker="s", s=90, zorder=5),
    "teacher":  dict(color="gray", marker="^", s=90, zorder=5),
    "kd":       dict(color="tab:orange", marker="D", s=80, zorder=4),
    "pruning":  dict(color="tab:blue", marker="o", s=80, zorder=4),
}


def match_experiment(weights_path):
    for key, (group, label, offset) in EXPERIMENT_INFO.items():
        if key in weights_path:
            return group, label, offset
    return "other", os.path.basename(os.path.dirname(os.path.dirname(weights_path))), (8, 8)


def load_results(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for r in reader:
            group, label, offset = match_experiment(r["weights"])
            rows.append({
                "label": label,
                "group": group,
                "offset": offset,
                "mAP50": float(r["mAP50"]),
                "fps": float(r["fps"]),
                "params_M": float(r["params_M"]),
            })
    return rows


def plot_tradeoff(rows, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 11))

    # ---------- 왼쪽: 파라미터 수 vs mAP ----------
    ax = axes[0]
    pruning_rows = sorted([r for r in rows if r["group"] == "pruning"], key=lambda r: r["params_M"])
    if pruning_rows:
        ax.plot([r["params_M"] for r in pruning_rows],
                 [r["mAP50"] for r in pruning_rows],
                 color=GROUP_STYLE["pruning"]["color"], linestyle="--", linewidth=1.5, zorder=3)

    seen_groups = set()
    for r in rows:
        style = GROUP_STYLE.get(r["group"], dict(color="tab:red", marker="x", s=70, zorder=4))
        # 범례에는 그룹당 한 번만 등록되도록 처리 (아래 legend 구성에서 별도 처리)
        ax.scatter(r["params_M"], r["mAP50"], label=r["label"], **style)

    for r in rows:
        if r["group"] in ("baseline", "teacher"):
            # 요청: 위쪽(파라미터) 그래프에서는 baseline/teacher 라벨을 점 왼쪽에 배치
            offset = (-r["offset"][0], r["offset"][1])
            ha = "right"
        else:
            offset = r["offset"]
            ha = "left"
        ax.annotate(r["label"], (r["params_M"], r["mAP50"]),
                    textcoords="offset points", xytext=offset, fontsize=8.5, ha=ha)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Model Size vs. Accuracy Trade-off")
    ax.grid(True, alpha=0.3, which="both")

    # ---------- 오른쪽: FPS vs mAP ----------
    ax2 = axes[1]
    for r in rows:
        style = GROUP_STYLE.get(r["group"], dict(color="tab:red", marker="x", s=70, zorder=4))
        ax2.scatter(r["fps"], r["mAP50"], **style)
    for r in rows:
        if r["group"] == "teacher":
            offset = r["offset"]
            ha = "left"
        elif r["group"] == "baseline":
            offset = (-r["offset"][0], r["offset"][1])
            ha = "right"
        else:
            offset = (-r["offset"][0], r["offset"][1])
            ha = "right"
        ax2.annotate(r["label"], (r["fps"], r["mAP50"]),
                     textcoords="offset points", xytext=offset, fontsize=8.5, ha=ha)

    ax2.set_xlabel("FPS (batch=1, 640px)")
    ax2.set_ylabel("mAP@0.5")
    ax2.set_title("Inference Speed vs. Accuracy Trade-off")
    ax2.grid(True, alpha=0.3)

    # ---------- 범례: 그래프 바깥 아래쪽에 한 번만 표시 (그래프 안 가림) ----------
    # 그룹별 대표 마커 하나씩만 모아서 커스텀 legend 구성
    from matplotlib.lines import Line2D
    group_order = ["baseline", "teacher", "kd", "pruning"]
    group_display = {
        "baseline": "Baseline (YOLOv8n)",
        "teacher": "Teacher (YOLOv8l)",
        "kd": "Knowledge Distillation",
        "pruning": "Structured Pruning",
    }
    legend_elems = []
    for g in group_order:
        if any(r["group"] == g for r in rows):
            st = GROUP_STYLE[g]
            legend_elems.append(Line2D([0], [0], marker=st["marker"], color="w",
                                        markerfacecolor=st["color"], markersize=9,
                                        label=group_display[g]))

    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle("Lightweight Object Detection: Accuracy vs. Efficiency", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])  # 아래쪽에 범례 공간 확보 -> 그래프 안 가림

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../outputs/results_summary.csv")
    parser.add_argument("--output", type=str, default="../outputs/figures/tradeoff.png")
    args = parser.parse_args()

    rows = load_results(args.csv)
    if not rows:
        print("결과가 없습니다. CSV 경로를 확인하세요.")
        return
    plot_tradeoff(rows, args.output)


if __name__ == "__main__":
    main()

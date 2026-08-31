"""
plot_results.py(Params/FPS 트레이드오프)에 이어, 추가 비교 그래프 3종을 생성.
1) Accuracy vs Model Size(MB)
2) Radar chart (정확도/속도/경량화를 정규화하여 종합 비교)
3) Pareto curve (Params vs mAP, 파레토 최적점 강조)

사용:
    python plot_extra.py --csv ../outputs/results_summary.csv --outdir ../outputs/figures
"""
import argparse
import os
import csv as csv_module
import numpy as np
import matplotlib.pyplot as plt

from plot_results import EXPERIMENT_INFO, GROUP_STYLE, match_experiment, load_results


def label_offset(r):
    """baseline/teacher는 왼쪽 정렬, KD는 baseline과 안 겹치게 아래쪽, 나머지는 기본(오른쪽)."""
    if r["group"] == "teacher":
        return (-8, -4), "right"
    if r["group"] == "baseline":
        return (8, 10), "left"
    if r["group"] == "kd":
        return (8, -16), "left"
    return (6, 6), "left"


def plot_size_vs_map(rows, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        style = GROUP_STYLE.get(r["group"], dict(color="tab:red", marker="x", s=70, zorder=4))
        ax.scatter(r.get("size_MB", 0), r["mAP50"], **style)
        offset, ha = label_offset(r)
        ax.annotate(r["label"], (r.get("size_MB", 0), r["mAP50"]),
                    textcoords="offset points", xytext=offset, fontsize=8, ha=ha)
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Model Size vs. Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"저장 완료: {outpath}")


def plot_radar(rows, outpath):
    # 비교 대상: baseline, teacher, 각 pruning ratio (KD는 baseline과 거의 동일하므로 생략)
    targets = [r for r in rows if r["group"] in ("baseline", "teacher", "pruning")]

    # 정규화 기준(0~1): mAP는 그대로, FPS/파라미터(작을수록 좋음->역수)는 최대값 기준 정규화
    max_map = max(r["mAP50"] for r in targets)
    max_fps = max(r["fps"] for r in targets)
    min_params = min(r["params_M"] for r in targets)

    metrics = ["mAP@0.5", "FPS", "Efficiency (1/Params)"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # pruning ratio별로 색을 다르게 (겉보기에 모두 같은 파란색이 되지 않도록)
    PRUNING_COLORS = {
        "Pruning (ratio=0.15)": "tab:blue",
        "Pruning (ratio=0.30)": "tab:green",
        "Pruning (ratio=0.45)": "tab:purple",
    }

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for r in targets:
        values = [
            r["mAP50"] / max_map,
            r["fps"] / max_fps,
            min_params / r["params_M"],
        ]
        values += values[:1]
        color = PRUNING_COLORS.get(r["label"], GROUP_STYLE.get(r["group"], {}).get("color", "tab:red"))
        ax.plot(angles, values, color=color, linewidth=2, label=r["label"])
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.tick_params(axis="x", pad=15)  # mAP@0.5 라벨이 점에 너무 붙지 않도록 살짝 띄움
    ax.set_yticklabels([])
    ax.set_title("Normalized Multi-metric Comparison (Radar Chart)", pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"저장 완료: {outpath}")


def pareto_front(points):
    """points: list of (x, y) where x=params(작을수록 좋음), y=mAP(클수록 좋음).
    반환: 파레토 최적점 인덱스 집합."""
    pareto_idx = set()
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if j == i:
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            pareto_idx.add(i)
    return pareto_idx


SHORT_LABEL = {
    "YOLOv8n (Baseline)": "Base",
    "YOLOv8l (Teacher)": "Teacher",
    "KD (weight=1.0)": "KD1.0",
    "KD (weight=0.3)": "KD0.3",
    "Pruning (ratio=0.15)": "P15",
    "Pruning (ratio=0.30)": "P30",
    "Pruning (ratio=0.45)": "P45",
}

SHORT_OFFSET = {
    "Base": (-18, 14),
    "KD1.0": (22, 6),
    "KD0.3": (0, -16),
}


def plot_pareto(rows, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    points = [(r["params_M"], r["mAP50"]) for r in rows]
    pareto_idx = pareto_front(points)

    for i, r in enumerate(rows):
        style = GROUP_STYLE.get(r["group"], dict(color="tab:red", marker="x", s=70, zorder=4)).copy()
        is_pareto = i in pareto_idx
        style["s"] = style.get("s", 70) * (1.3 if is_pareto else 1.0)
        ax.scatter(r["params_M"], r["mAP50"],
                   edgecolors="red" if is_pareto else "none",
                   linewidths=2 if is_pareto else 0, **style)

        # 점 위에 짧은 코드(Base/Teacher/KD1.0/P15 등) 바로 표시
        short = SHORT_LABEL.get(r["label"], r["label"])
        short_offset = SHORT_OFFSET.get(short, (0, 12))
        ax.annotate(short, (r["params_M"], r["mAP50"]),
                    textcoords="offset points", xytext=short_offset, fontsize=8,
                    fontweight="bold", ha="center", color="black")

        label = r["label"] + ("  ★" if is_pareto else "")
        offset, ha = label_offset(r)
        # 짧은 코드와 겹치지 않도록 전체 설명 라벨은 조금 더 멀리 배치
        offset = (offset[0], offset[1] + (14 if offset[1] >= 0 else -14))
        ax.annotate(label, (r["params_M"], r["mAP50"]),
                    textcoords="offset points", xytext=offset, fontsize=7.5, ha=ha, color="dimgray")

    # 파레토 프론트를 선으로 연결 (params 오름차순)
    pareto_points = sorted([points[i] for i in pareto_idx])
    if len(pareto_points) > 1:
        ax.plot([p[0] for p in pareto_points], [p[1] for p in pareto_points],
                color="red", linestyle=":", linewidth=1.5, zorder=2, label="Pareto Frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Pareto Curve: Parameters vs. Accuracy\n(★ = Pareto-optimal)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"저장 완료: {outpath}")


def load_results_with_size(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for r in reader:
            group, label, offset = match_experiment(r["weights"])
            size_val = r.get("size_MB", "").strip()
            rows.append({
                "label": label,
                "group": group,
                "mAP50": float(r["mAP50"]),
                "fps": float(r["fps"]),
                "params_M": float(r["params_M"]),
                "size_MB": float(size_val) if size_val not in ("", "-") else 0.0,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../outputs/results_summary.csv")
    parser.add_argument("--outdir", type=str, default="../outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_results_with_size(args.csv)
    if not rows:
        print("결과가 없습니다.")
        return

    plot_size_vs_map(rows, os.path.join(args.outdir, "size_vs_map.png"))
    plot_radar(rows, os.path.join(args.outdir, "radar.png"))
    plot_pareto(rows, os.path.join(args.outdir, "pareto.png"))


if __name__ == "__main__":
    main()

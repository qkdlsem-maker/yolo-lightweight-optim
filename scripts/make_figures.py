"""
v2: 이전 버전의 문제 수정.
  1) 레이블 텍스트 겹침 -> adjustText로 자동 배치 + 지시선(leader line) 연결
  2) KD 3종(w=1.0, w=0.3, TAKD)이 같은 색/마커라 구분 안 됨 -> 10개 모델 전부 고유 색상 부여
     (마커 모양은 기존처럼 "기법 카테고리"를 나타내고, 색상은 "개별 모델"을 나타내도록 이중 인코딩)

설치:
    pip install matplotlib numpy adjustText

사용:
    python make_figures.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

OUT_DIR = "../outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 최종 확정 데이터 ----------------
# (label, params_M, size_MB, mAP50, mAP50_95, fps_mean, fps_std, category)
DATA = [
    ("YOLOv8n (Baseline)",        3.01, 5.94,  0.463, 0.260, 367.2, 95.2,  "baseline"),
    ("YOLOv8l (Teacher)",        43.61, 83.58, 0.583, 0.342, 164.2, 5.1,   "teacher"),
    ("KD (w=1.0, YOLOv8l)",       3.01, 5.86,  0.460, 0.258, 598.2, 197.4, "kd"),
    ("KD (w=0.3, YOLOv8l)",       3.01, 5.90,  0.460, None,  343.7, 66.8,  "kd"),
    ("TAKD (YOLOv8s teacher)",    3.01, 5.94,  0.460, None,  None,  None,  "kd"),
    ("Pruning (r=0.15)",          2.28, 4.61,  0.424, 0.233, 301.2, 58.7,  "pruning"),
    ("Pruning (r=0.30)",          1.68, 3.34,  0.380, 0.209, 477.5, 191.3, "pruning"),
    ("Pruning (r=0.45)",          1.19, 2.53,  0.371, 0.202, 322.0, 94.7,  "pruning"),
    ("Pruning (0.45) + KD",       1.19, 2.53,  0.367, 0.200, 326.9, 91.3,  "pruning_kd"),
    ("Pruning (0.45) + FP16",     1.19, 2.53,  0.370, None,  202.9, 43.3,  "pruning_fp16"),
]

CATEGORY_MARKER = {
    "baseline": "s", "teacher": "^", "kd": "v",
    "pruning": "o", "pruning_kd": "D", "pruning_fp16": "P",
}
CATEGORY_LABEL = {
    "baseline": "Baseline", "teacher": "Teacher (YOLOv8l)", "kd": "Knowledge Distillation",
    "pruning": "Structured Pruning", "pruning_kd": "Pruning + KD", "pruning_fp16": "Pruning + FP16 PTQ",
}
MODEL_COLORS = plt.cm.tab10(np.linspace(0, 1, len(DATA)))


def _scatter_with_labels(ax, xkey_idx, ax_title, ax_xlabel, xscale=None):
    texts = []
    for i, row in enumerate(DATA):
        label, params, size, map50 = row[0], row[1], row[2], row[3]
        x = row[xkey_idx]
        cat = row[7]
        color = MODEL_COLORS[i]
        marker = CATEGORY_MARKER[cat]
        ax.scatter(x, map50, s=110, c=[color], marker=marker,
                   edgecolors="black", linewidths=0.6, zorder=3)
        t = ax.text(x, map50, f"  {label}", fontsize=8, zorder=4)
        texts.append(t)

    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(ax_xlabel)
    ax.set_ylabel("mAP@0.5")
    ax.set_title(ax_title)
    ax.grid(True, alpha=0.3)

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
        expand_points=(1.4, 1.6),
        force_text=(0.6, 0.9),
        force_points=(0.3, 0.5),
    )

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="lightgray",
                       markeredgecolor="black", markersize=10, label=CATEGORY_LABEL[cat])
               for cat, m in CATEGORY_MARKER.items()]
    ax.legend(handles=handles, loc="best", fontsize=8, title="Marker shape = method",
              title_fontsize=8, framealpha=0.9)


def fig1_params_fps_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _scatter_with_labels(ax, xkey_idx=1, ax_title="mAP@0.5 vs. Number of Parameters",
                          ax_xlabel="Number of Parameters (M, log scale)", xscale="log")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_params_fps_tradeoff.png", dpi=200)
    plt.close(fig)


def fig2_size_vs_accuracy():
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _scatter_with_labels(ax, xkey_idx=2, ax_title="Model Size vs. Accuracy",
                          ax_xlabel="Model Size (MB)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_size_vs_accuracy.png", dpi=200)
    plt.close(fig)


def fig3_radar_multi_metric():
    selected_labels = [
        "YOLOv8n (Baseline)", "KD (w=1.0, YOLOv8l)",
        "Pruning (r=0.45)", "Pruning (0.45) + KD",
    ]
    rows = [d for d in DATA if d[0] in selected_labels]

    metrics = ["mAP@0.5", "1/Params", "1/Size", "FPS"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    raw = np.array([
        [r[3], 1 / r[1], 1 / r[2], (r[5] if r[5] is not None else np.nan)]
        for r in rows
    ], dtype=float)
    col_max = np.nanmax(raw, axis=0)
    col_min = np.nanmin(raw, axis=0)
    norm = (raw - col_min) / (col_max - col_min + 1e-9)
    norm = np.nan_to_num(norm, nan=0.0)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    colors = ["black", "tab:orange", "tab:blue", "tab:green"]
    for i, row in enumerate(rows):
        label = row[0]
        values = norm[i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=label)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Normalized Multi-Metric Comparison (Radar Chart)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_radar_multi_metric.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_pareto_frontier():
    fig, ax = plt.subplots(figsize=(8, 6.5))

    pts = [(d[1], d[3]) for d in DATA]
    pareto = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    pareto.sort(key=lambda x: x[0])

    _scatter_with_labels(ax, xkey_idx=1, ax_title="Pareto Frontier: mAP@0.5 vs. Number of Parameters",
                          ax_xlabel="Number of Parameters (M, log scale)", xscale="log")

    if len(pareto) > 1:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, "r--", linewidth=1.5, alpha=0.7, zorder=1, label="Pareto Frontier")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_pareto_frontier.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_params_fps_tradeoff()
    fig2_size_vs_accuracy()
    fig3_radar_multi_metric()
    fig4_pareto_frontier()
    print(f"4개 그림 저장 완료: {OUT_DIR}/")

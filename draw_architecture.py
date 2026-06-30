#!/usr/bin/env python
"""Generate a system architecture diagram for the Good-Badminton project."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=9):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1", facecolor=color,
        edgecolor="white", linewidth=1.5, alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(0, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color="#888888"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(-9, 9)
ax.set_ylim(-4.5, 6.0)
ax.set_aspect("equal")
ax.axis("off")

# ── Colours ──────────────────────────────────────────────────────
C_INPUT   = "#E67E22"
C_CORE    = "#2980B9"
C_DETECT  = "#27AE60"
C_MODELS  = "#8E44AD"
C_TRACK   = "#16A085"
C_VIS     = "#D35400"
C_OUTPUT  = "#C0392B"
BG        = "#1a1a2e"
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

# ── Title ─────────────────────────────────────────────────────────
ax.text(0, 5.5, "Good‑Badminton  系统架构", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")

# ── Input Layer ───────────────────────────────────────────────────
y_in = 4.3
draw_box(ax, -6.5, y_in, 2.6, 0.9, "📹 视频输入\nvideos/*.mp4", C_INPUT)
draw_box(ax, -3.0, y_in, 2.6, 0.9, "🖼️ 球场模板\ncourt template", C_INPUT)
draw_box(ax,  0.5, y_in, 2.6, 0.9, "⚙️ CLI 参数\nargparse", C_INPUT)
draw_box(ax,  4.0, y_in, 2.6, 0.9, "📦 模型权重\nQNN / ONNX / PT", C_MODELS)

# ── Core System ───────────────────────────────────────────────────
y_core = 2.8
draw_box(ax, 0, y_core, 5.5, 1.2,
         "🔧 BadmintonAnalysisSystem\n( system.py )", C_CORE, fontsize=11)

# Inputs → Core
for xx in [-6.5, -3.0, 0.5, 4.0]:
    arrow(ax, xx, y_in - 0.45, 0, y_core + 0.6, "#88888877")

# ── Detection Modules ─────────────────────────────────────────────
y_det = 1.3
det_info = [
    (-4.8, "🤸 姿态检测\nYolo11PoseQNNProcessor\n(QNN / NPU)", C_DETECT),
    (0.0,  "🏸 羽毛球检测\nYolo11sBallDetector\n(QNN / NPU)", C_DETECT),
    (4.8,  "🏃 人体检测\nYOLO / RTMPose\n(PT / ONNX)", C_DETECT),
]
for xx, label, c in det_info:
    draw_box(ax, xx, y_det, 4.2, 1.2, label, c, fontsize=8)

arrow(ax, -1.5, y_core - 0.6, -4.8, y_det + 0.6)
arrow(ax, 0,    y_core - 0.6, 0,    y_det + 0.6)
arrow(ax, 1.5,  y_core - 0.6, 4.8,  y_det + 0.6)

# ── Tracking & Mapping ────────────────────────────────────────────
y_trk = -0.4
track_info = [
    (-3.5, "👤 球员追踪\nPlayerTracker", C_TRACK),
    (3.5,  "🏸 羽毛球追踪\nShuttlecockTracker", C_TRACK),
]
for xx, label, c in track_info:
    draw_box(ax, xx, y_trk, 3.2, 1.0, label, c, fontsize=8)

draw_box(ax, 0, y_trk - 1.5, 3.8, 1.0,
         "🗺️ 球场映射\nCourtMapper", C_TRACK, fontsize=8)

arrow(ax, -4.8, y_det - 0.6, -3.5, y_trk + 0.5)
arrow(ax, 0,    y_det - 0.6, 3.5,  y_trk + 0.5)
arrow(ax, -3.5, y_trk - 0.5, -1.5, y_trk - 1.0, "#16A08577")
arrow(ax, 3.5,  y_trk - 0.5, 1.5,  y_trk - 1.0, "#16A08577")

# ── Visualization ─────────────────────────────────────────────────
y_vis = -2.8
vis_info = [
    (-5.0, "🦴 姿态绘制\nPlayerPoseVisualizer", C_VIS),
    (-1.2, "📊 统计面板\nStatsVisualizer", C_VIS),
    (2.6, "🏟️ 球场轨迹\nCourtTrajectoryVisualizer", C_VIS),
]
for xx, label, c in vis_info:
    draw_box(ax, xx, y_vis, 3.3, 1.0, label, c, fontsize=7.5)

for xx, _, _ in vis_info:
    arrow(ax, 0, y_trk - 2.0, xx, y_vis + 0.5, "#88888877")

# ── Output Layer ──────────────────────────────────────────────────
y_out = -4.0
out_info = [
    (-3.5, "🎬 标注视频\n.mp4", C_OUTPUT),
    (0.0,  "📝 检测日志\n.jsonl", C_OUTPUT),
    (3.5,  "📈 可视化\n热力图 / 散点图", C_OUTPUT),
]
for xx, label, c in out_info:
    draw_box(ax, xx, y_out, 3.0, 0.9, label, c, fontsize=8)
    arrow(ax, xx, y_vis - 0.5, xx, y_out + 0.45, "#88888877")

# ── Legend ────────────────────────────────────────────────────────
legend_y = -4.45
legend_items = [
    ("Input / CLI", C_INPUT), ("Core System", C_CORE),
    ("Detection", C_DETECT), ("Models", C_MODELS),
    ("Tracking / Mapping", C_TRACK), ("Visualization", C_VIS),
    ("Output", C_OUTPUT),
]
for i, (lab, col) in enumerate(legend_items):
    lx = -8 + i * 2.6
    ax.add_patch(plt.Rectangle((lx, legend_y), 0.35, 0.35, color=col,
                                transform=ax.transData, clip_on=False))
    ax.text(lx + 0.5, legend_y + 0.17, lab, color="white",
            fontsize=7, va="center")

out_path = "/home/aidlux/2026_6_25/Good-Badminton-main/architecture.png"
plt.tight_layout(pad=1)
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Architecture diagram saved to: {out_path}")

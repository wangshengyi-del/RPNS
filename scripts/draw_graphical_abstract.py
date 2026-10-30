#!/usr/bin/env python3
"""ASOC graphical abstract: 5 x 13 cm banner, vector PDF + 300 dpi PNG/TIFF.

Drawn programmatically (matplotlib), not with a generative image model.
Elsevier minimum is 531 x 1328 px; this export is larger and still readable
when shown at 5 x 13 cm / 96 dpi.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle,
)

OUT = Path(__file__).resolve().parent

# Same family as the RPNS architecture figure
C_TASK = "#B36B1A"
C_PROC_C = "#5B9BD5"
C_PROC_E = "#70AD47"
C_RULE = "#2F7D4F"
C_LEARN = "#2B6CB0"
C_RESULT = "#C05621"
C_EDGE = "#2F3A4A"
C_MUTED = "#5C6773"
C_LINE = "#3D4A5C"
C_TITLE = "#1B2838"
BG = "#F7F4EE"
P1 = "#F4E6C8"
P2 = "#D7E8D4"
P3 = "#D4E3F2"
P4 = "#F3D9C4"


def _font():
    for name in ("Helvetica Neue", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


FONT = _font()


def rbox(ax, x, y, w, h, fc, ec=C_LINE, lw=1.6, r=0.12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.015,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        )
    )


def txt(ax, x, y, s, size=14, weight="medium", color=C_EDGE,
        ha="center", va="center", style="normal"):
    ax.text(
        x, y, s, fontsize=size, fontweight=weight, color=color,
        ha=ha, va=va, zorder=6, linespacing=1.15,
        fontfamily=FONT, fontstyle=style,
    )


def arrow(ax, x1, y1, x2, y2, color=C_LINE):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=18, lw=2.0,
            color=color, zorder=3, shrinkA=0, shrinkB=0,
        )
    )


def mini_dag(ax, cx, cy, s=0.42, color=C_TASK, fill="#F8E3B0", kind="fork"):
    if kind == "join":
        pts = [
            (cx - 0.58 * s, cy + 0.42 * s),
            (cx + 0.58 * s, cy + 0.42 * s),
            (cx, cy - 0.05 * s),
            (cx - 0.28 * s, cy - 0.48 * s),
            (cx + 0.32 * s, cy - 0.48 * s),
        ]
        edges = ((0, 2), (1, 2), (2, 3), (2, 4))
    else:
        pts = [
            (cx - 0.55 * s, cy + 0.48 * s),
            (cx + 0.38 * s, cy + 0.55 * s),
            (cx - 0.62 * s, cy - 0.32 * s),
            (cx + 0.18 * s, cy - 0.12 * s),
            (cx + 0.70 * s, cy + 0.06 * s),
        ]
        edges = ((0, 2), (0, 3), (1, 3), (1, 4), (3, 4))
    for i, j in edges:
        ax.plot(
            [pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
            color=color, lw=1.7, zorder=5, solid_capstyle="round",
        )
    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.070, fc=fill, ec=color, lw=1.5, zorder=6))


def job_card(ax, x, y, w, h, color, fill, label, kind="fork"):
    rbox(ax, x, y, w, h, "white", ec=color, lw=1.5, r=0.08)
    mini_dag(ax, x + w / 2, y + h / 2 + 0.12, s=0.50, color=color, fill=fill, kind=kind)
    txt(ax, x + w / 2, y + 0.20, label, size=12, weight="bold", color=color)


def icon_cloud(ax, cx, cy, s=1.0, fc=C_PROC_C):
    """Cloud silhouette: overlapping blobs, outline then fill."""
    def blobs(k, color, z):
        patches = [
            Circle((cx - 0.22 * k, cy - 0.02 * k), 0.20 * k),
            Circle((cx + 0.02 * k, cy + 0.12 * k), 0.24 * k),
            Circle((cx + 0.24 * k, cy + 0.00 * k), 0.18 * k),
            Ellipse((cx, cy - 0.12 * k), 0.78 * k, 0.32 * k),
        ]
        for p in patches:
            p.set_facecolor(color)
            p.set_edgecolor("none")
            p.set_zorder(z)
            ax.add_patch(p)

    blobs(s * 1.10, C_LINE, 5)
    blobs(s, fc, 6)


def icon_server(ax, cx, cy, s=1.0, fc=C_PROC_E):
    """Edge processor as a small rack with drive bays."""
    w, h = 0.40 * s, 0.62 * s
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=fc, edgecolor=C_LINE, linewidth=1.4, zorder=5,
        )
    )
    bay_h = h * 0.16
    gap = h * 0.07
    top = y0 + h - 0.08 * s
    for i in range(3):
        yy = top - i * (bay_h + gap) - bay_h
        ax.add_patch(Rectangle(
            (x0 + 0.07 * s, yy), w - 0.14 * s, bay_h,
            fc="#F4F7FA", ec="none", zorder=6,
        ))
        ax.add_patch(Circle(
            (x0 + w - 0.11 * s, yy + bay_h / 2), 0.028 * s,
            fc="#F4E27C", ec="none", zorder=7,
        ))


def draw():
    # ~2.6x the 5 x 13 cm viewing size so type stays readable after scale-down
    fig_w, fig_h = 13.28, 5.31
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.add_patch(Rectangle((0, fig_h - 0.10), fig_w, 0.10, fc=C_TITLE, ec="none", zorder=1))

    txt(ax, 6.64, 4.95, "RPNS: a rule-prior neural scheduling agent",
        size=22, weight="bold", color=C_TITLE)
    txt(ax, 6.64, 4.52, "Deadline-constrained multi-workflow allocation in cloud–edge systems",
        size=13, color=C_MUTED)

    panels = [
        (0.20, 0.22, 2.88, 3.98, P1, "1   Setting"),
        (3.40, 0.22, 2.88, 3.98, P2, "2   Rule prior"),
        (6.60, 0.22, 2.88, 3.98, P3, "3   Neural placement"),
        (9.80, 0.22, 3.28, 3.98, P4, "4   Result"),
    ]
    for x, y, w, h, fc, title in panels:
        rbox(ax, x, y, w, h, fc)
        txt(ax, x + 0.18, y + h - 0.36, title, size=15, weight="bold", color=C_TITLE, ha="left")

    for x1, x2 in ((3.08, 3.40), (6.28, 6.60), (9.48, 9.80)):
        arrow(ax, x1, 2.20, x2, 2.20)

    # Panel 1: arriving jobs above a shared cloud–edge pool
    job_card(ax, 0.38, 2.38, 1.22, 1.32, C_TASK, "#F8E3B0", "Job A", kind="fork")
    job_card(ax, 1.68, 2.38, 1.22, 1.32, "#8A5A12", "#EBD3A8", "Job B", kind="join")
    txt(ax, 1.64, 2.18, "Jobs arrive online", size=12, color=C_MUTED)

    rbox(ax, 0.38, 0.72, 2.52, 1.32, "white", r=0.08)
    icon_cloud(ax, 0.95, 1.50, s=0.82)
    icon_server(ax, 2.28, 1.50, s=1.08)
    txt(ax, 0.95, 0.92, "Cloud", size=12, weight="bold", color=C_LEARN)
    txt(ax, 2.28, 0.92, "Edge", size=12, weight="bold", color=C_RULE)
    txt(ax, 1.64, 0.48, "Shared processors", size=11, color=C_MUTED)

    # Panel 2
    rows = [("v2", "LFT = 7", True), ("v3", "LFT = 11", False), ("v1", "LFT = 12", False)]
    y0 = 3.12
    for i, (name, lft, hot) in enumerate(rows):
        yy = y0 - i * 0.62
        fc = C_RULE if hot else "white"
        tc = "white" if hot else C_EDGE
        rbox(ax, 3.64, yy - 0.22, 2.40, 0.48, fc, ec=C_RULE if hot else C_LINE, lw=1.4, r=0.08)
        txt(ax, 4.84, yy + 0.02, f"{name}     {lft}",
            size=14, weight="bold" if hot else "medium", color=tc)
    txt(ax, 4.84, 1.12, r"$v^{*}=\arg\min$ LFT", size=15, weight="bold", color=C_RULE)
    txt(ax, 4.84, 0.62, "EDF picks the urgent task", size=13, color=C_MUTED)

    # Panel 3: same level — encoder name vs policy name; verbs in the subtitles
    rbox(ax, 6.86, 2.52, 2.36, 0.98, "white", r=0.08)
    txt(ax, 8.04, 3.18, "Hierarchical GAT", size=14, weight="bold", color=C_LEARN)
    txt(ax, 8.04, 2.78, "encode multi-job state", size=13, color=C_MUTED)
    rbox(ax, 6.86, 1.38, 2.36, 0.98, "white", r=0.08)
    txt(ax, 8.04, 2.04, "PPO", size=14, weight="bold", color=C_LEARN)
    txt(ax, 8.04, 1.64, "select the processor", size=13, color=C_MUTED)
    txt(ax, 8.04, 0.62, "Learn processor selection only", size=13, color=C_MUTED)

    # Panel 4
    txt(ax, 11.44, 3.18, "Job on-time rate", size=14, color=C_MUTED)
    txt(ax, 11.44, 2.62, r"$81.54\%$", size=28, weight="bold", color=C_RESULT)
    txt(ax, 11.44, 2.08, r"$+12.5$ pp vs EDF", size=14, weight="bold", color=C_RESULT)
    txt(ax, 11.44, 1.68, r"$+6.8$ pp vs end-to-end PPO", size=14, weight="bold", color=C_RESULT)
    txt(ax, 11.44, 1.18, "Lowest average energy", size=13, weight="medium")
    txt(ax, 11.44, 0.68, "Better than EDF on Pegasus", size=12, color=C_MUTED)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf = OUT / "graphical_abstract.pdf"
    png = OUT / "graphical_abstract.png"
    tif = OUT / "graphical_abstract.tiff"
    fig.savefig(pdf, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(tif, dpi=300, facecolor=fig.get_facecolor(), pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    # Elsevier portals prefer opaque RGB rasters.
    from PIL import Image
    for path in (png, tif):
        im = Image.open(path).convert("RGB")
        im.save(path)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"wrote {tif}")


if __name__ == "__main__":
    draw()

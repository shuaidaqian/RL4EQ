"""中期答辩材料的统一视觉规范。

配色、字体、字号集中在此，图与 PPT 共用，保证整套材料看起来是"一个系统"。
"""

from pathlib import Path

# ---------- 路径 ----------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # 仓库根
RESULTS = REPO / "docs" / "results"
FIGDIR = HERE / "figures"
LEGACY_ASSETS = Path("D:/Research/RL4EQ/output/midterm_defense/assets")

# ---------- 色板 ----------
INK      = "#1B2A3A"   # 主文字：近黑墨蓝
MUTED    = "#6B7A89"   # 次要文字 / 轴线
GRID     = "#DCD6CC"   # 网格线
BG       = "#FFFFFF"   # 白色背景（与论文插图统一）
PANEL    = "#F2F4F7"   # 浅灰块

NAVY     = "#1B3A57"   # 主色：深靛蓝
BLUE     = "#3E7CB1"   # 次色：中蓝
FROZEN   = "#7A8FA6"   # Frozen / 基线：冷灰蓝
BAYES    = "#D98E30"   # 本文方法：琥珀
TEAL     = "#2E6F8E"   # 辅助：青
ALERT    = "#B4472F"   # 警示：砖红
GOOD     = "#4F7C5B"   # 通过：墨绿

# ---------- PPT 版式 ----------
SLIDE_W, SLIDE_H = 13.333, 7.5      # 16:9，单位英寸
MARGIN_X = 0.72
TITLE_Y = 0.42
BODY_TOP = 1.62
FOOT_Y = 6.98

FONT_CJK = "微软雅黑"                # 系统自带，答辩机器上不会缺字
FONT_LATIN = "Segoe UI"

# ---------- matplotlib 全局样式 ----------


def apply_matplotlib_style():
    """干净学术风：细轴线、无上右边框、朝内小刻度、无图例边框。"""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"],
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })
    return plt

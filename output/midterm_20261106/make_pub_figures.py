"""第二研究点期刊级图（Python/matplotlib）。

Figure contract
---------------
核心结论：在 116 符号长记忆 EME 信道上，"离线训练 + 在线贝叶斯 CIR 适配"
相对同源冻结网络获得可重复、可归因、随帧累积的 Data BER 增益。
证据链：Fig1 离线均衡能力（含公平对照）→ Fig2 独立确认主结果 →
Fig3 机制（增益演化、状态估计精度、规模稳健性）。
原型：quantitative grid，每图一个 hero panel。
导出：180 mm 双栏，Arial 7 pt，SVG/PDF（矢量）+ PNG 600 dpi。
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "docs" / "results"
OUT = HERE / "figures" / "pub"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 期刊样式 ----------
PAL = {
    "frozen": "#767676",
    "bayes": "#0F4D92",
    "trad1": "#A8A8A8",
    "trad2": "#767676",
    "trad3": "#CFCECE",
    "up": "#2E9E44",
    "down": "#B64342",
    "ink": "#272727",
    "grid": "#E3E3E3",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": PAL["ink"],
    "xtick.color": PAL["ink"],
    "ytick.color": PAL["ink"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "grid.color": PAL["grid"],
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MM = 1 / 25.4
W2 = 180 * MM          # 双栏宽
SNRS = [0, 5, 10, 15]
X = np.arange(4)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  ✓ {name}")


def load():
    d = {}
    d["conf"] = json.loads((RES / "bayesian_cir_confirmation_v1_20261106/summary.json").read_text(encoding="utf-8"))
    d["rows"] = json.loads((RES / "bayesian_cir_confirmation_v1_20261106/confirmation.json").read_text(encoding="utf-8"))["rows"]
    for k, f in [("5x32", "summary_5x32.json"), ("10x64", "summary_10x64.json"), ("10x128", "summary_10x128.json")]:
        d[k] = json.loads((RES / "bayesian_cir_diag_v1_20261106" / f).read_text(encoding="utf-8"))
    return d


def panel_tag(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=PAL["ink"])


# ---------------------------------------------------------------- Fig 1 离线能力
def fig_offline():
    nn = [22.1947, 5.9599, 1.0598, 0.2156]
    dfe = [44.0524, 18.2986, 6.9090, 4.3743]
    lm = [44.0054, 22.4306, 7.2229, 5.0456]

    fair = {
        "Bayes": [0.181424, 0.049344, 0.007270, 0.001383],
        "Frozen": [0.214572, 0.052544, 0.008138, 0.001519],
        "DFE-RLS": [0.406359, 0.170546, 0.035862, 0.017442],
        "LMMSE-FIR": [0.436551, 0.149495, 0.041314, 0.016710],
        "SC-FDE": [0.490641, 0.485677, 0.484185, 0.482829],
    }

    fig = plt.figure(figsize=(W2, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 0.85], wspace=0.42)

    # a: 主矩阵 BER 曲线
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(X, dfe, "s--", color=PAL["trad2"], ms=3, label="DFE-RLS")
    ax.semilogy(X, lm, "^:", color=PAL["trad1"], ms=3, label="LMMSE-FIR")
    ax.semilogy(X, nn, "o-", color=PAL["bayes"], ms=4, label="Neural equalizer")
    for i in range(4):
        ax.annotate(f"−{(dfe[i]-nn[i])/dfe[i]*100:.0f}%", (X[i], nn[i]),
                    textcoords="offset points", xytext=(0, -11), ha="center",
                    fontsize=6, color=PAL["up"], fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.set_xticks(X); ax.set_xticklabels(SNRS)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Data BER")
    ax.set_title("Offline equalization", loc="left")
    ax.grid(True, axis="y", which="both", alpha=0.6)
    ax.legend(loc="lower left", handlelength=1.6)
    ax.set_ylim(0.1, 80)
    panel_tag(ax, "a")

    # b: 公平对照（3x16）
    ax = fig.add_subplot(gs[0, 1])
    order = [("Bayes", PAL["bayes"], "o"), ("Frozen", PAL["frozen"], "s"),
             ("DFE-RLS", "#767676", "^"), ("LMMSE-FIR", "#A8A8A8", "v"),
             ("SC-FDE", "#D0D0D0", "D")]
    for name, c, mk in order:
        ax.semilogy(X, fair[name], ls="none", marker=mk, color=c, ms=4.2,
                    label=name, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(8e-4, 1.6)
    ax.set_xticks(X); ax.set_xticklabels(SNRS)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Data BER")
    ax.set_title("Fair baseline comparison (3×16)", loc="left")
    ax.grid(True, axis="y", which="both", alpha=0.6)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              columnspacing=0.8, handletextpad=0.3, fontsize=6)
    panel_tag(ax, "b")

    # c: 倍数
    ax = fig.add_subplot(gs[0, 2])
    best = [min(fair["DFE-RLS"][i], fair["LMMSE-FIR"][i]) for i in range(4)]
    ratio = [best[i] / fair["Bayes"][i] for i in range(4)]
    bars = ax.bar(X, ratio, 0.55, color=PAL["bayes"], edgecolor="none")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, ratio[i] * 1.06, f"{ratio[i]:.1f}×",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=PAL["ink"])
    ax.set_xticks(X); ax.set_xticklabels(SNRS)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Best baseline / Bayes")
    ax.set_title("Relative gap", loc="left")
    ax.set_ylim(0, max(ratio) * 1.25)
    ax.grid(True, axis="y", alpha=0.6)
    panel_tag(ax, "c")

    save(fig, "fig1_offline")


# ---------------------------------------------------------------- Fig 2 在线确认
def fig_confirm(D):
    ps = D["conf"]["per_snr"]
    rows = D["rows"]
    frozen = [np.mean([r["frozen_ber"] for r in rows if r["snr_db"] == s]) for s in SNRS]
    bayes = [np.mean([r["bayes_ber"] for r in rows if r["snr_db"] == s]) for s in SNRS]
    delta = [ps[str(s)]["delta"] for s in SNRS]
    ci = [1.96 * ps[str(s)]["se"] for s in SNRS]
    deltas = [np.array(ps[str(s)]["deltas"]) for s in SNRS]

    fig = plt.figure(figsize=(W2, 96 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], wspace=0.34, hspace=0.55)

    # a: BER 曲线
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(X, frozen, "o-", color=PAL["frozen"], ms=4, label="Frozen (acquisition CIR)")
    ax.semilogy(X, bayes, "s-", color=PAL["bayes"], ms=4, label="Bayes CIR (online)")
    ax.set_xticks(X); ax.set_xticklabels(SNRS)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Data BER")
    ax.set_title("Confirmed BER", loc="left")
    ax.grid(True, axis="y", which="both", alpha=0.6)
    ax.legend(loc="lower left", handlelength=1.6)
    ax.set_xlim(-0.4, 3.5)
    panel_tag(ax, "a")

    # b: 森林图
    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(4)[::-1]
    ax.axvline(0, color=PAL["trad2"], lw=0.7, ls=(0, (3, 2)))
    ax.errorbar(delta, y, xerr=ci, fmt="D", color=PAL["bayes"], ecolor=PAL["bayes"],
                elinewidth=1.0, capsize=2.5, ms=4)
    for i in range(4):
        ax.annotate(f"+{delta[i]:.4f}", (delta[i], y[i]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=6.2, color=PAL["ink"])
    ax.set_yticks(y); ax.set_yticklabels([f"{s} dB" for s in SNRS])
    ax.set_xlabel("ΔBER = BER(Frozen) − BER(Bayes)")
    ax.set_title("Paired gain, 95% CI", loc="left")
    ax.grid(True, axis="x", alpha=0.6)
    ax.set_xlim(-0.002, max(delta) * 1.55)
    panel_tag(ax, "b")

    # c: 10 session strip
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(7)
    for i in range(4):
        v = deltas[i] * 1e3
        ax.scatter(np.full(10, i) + rng.uniform(-0.14, 0.14, 10), v, s=16,
                   color=PAL["bayes"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [delta[i] * 1e3] * 2, color=PAL["ink"], lw=1.2, zorder=4)
    ax.axhline(0, color=PAL["trad2"], lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{s} dB" for s in SNRS])
    ax.set_ylabel("Session ΔBER ×1000", labelpad=6)
    ax.set_title("Per-session gain (n = 10)", loc="left")
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_xlim(-0.5, 3.5)
    panel_tag(ax, "c")

    # d: 帧窗口演化
    ax = fig.add_subplot(gs[1, 1])
    h3 = D["conf"]["h3"]
    bins = np.arange(1, 9)
    colors = [PAL["bayes"], "#42949E", "#9A4D8E", "#B64342"]
    for s, c in zip(SNRS, colors):
        yv = np.array(h3[str(s)]["bins"]) * 1e3
        ax.plot(bins, yv, "o-", color=c, ms=3, lw=1.0, label=f"{s} dB")
    ax.set_yscale("log")
    ax.set_xlabel("Frame window (16 frames each)")
    ax.set_ylabel("Window ΔBER ×1000")
    ax.set_title("Gain across the session", loc="left")
    ax.set_xticks(bins)
    ax.grid(True, axis="y", which="both", alpha=0.5)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42),
              columnspacing=0.8, handlelength=1.2, fontsize=6)
    panel_tag(ax, "d")

    save(fig, "fig2_confirm")


# ---------------------------------------------------------------- Fig 3 机制
def fig_mechanism(D):
    ps = D["conf"]["per_snr"]
    fig = plt.figure(figsize=(W2, 58 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.05], wspace=0.42)

    # a: NMSE
    ax = fig.add_subplot(gs[0, 0])
    f = [ps[str(s)]["nmse_frozen"] for s in SNRS]
    b = [ps[str(s)]["nmse_bayes"] for s in SNRS]
    w = 0.34
    ax.bar(X - w / 2, f, w, color=PAL["frozen"], edgecolor="none", label="Frozen")
    ax.bar(X + w / 2, b, w, color=PAL["bayes"], edgecolor="none", label="Bayes")
    for i in range(4):
        ax.annotate(f"−{(f[i]-b[i])/f[i]*100:.0f}%", (X[i] + w / 2, b[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6, color=PAL["up"], fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.set_xticks(X); ax.set_xticklabels(SNRS)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("CIR NMSE")
    ax.set_title("State estimation", loc="left")
    ax.set_ylim(0, max(f) * 1.28)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="upper right", handlelength=1.0)
    panel_tag(ax, "a")

    # b: 规模一致性
    ax = fig.add_subplot(gs[0, 1])
    sc = [("5×32", D["5x32"]), ("10×64", D["10x64"]), ("10×128", D["10x128"])]
    xs = np.arange(3)
    for s, c in zip(SNRS, [PAL["bayes"], "#42949E", "#9A4D8E", "#B64342"]):
        g = np.array([d[str(s)]["gain"] for _, d in sc]) * 1e3
        ax.plot(xs, g, "o-", color=c, ms=3, lw=1.0, label=f"{s} dB")
    ax.axhline(0, color=PAL["trad2"], lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in sc])
    ax.set_xlabel("Development scale (seeds × frames)")
    ax.set_ylabel("ΔBER ×1000")
    ax.set_title("Scale consistency", loc="left")
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(ncol=2, loc="center left", bbox_to_anchor=(0.02, 0.52),
              columnspacing=0.8, handlelength=1.2, fontsize=6)
    panel_tag(ax, "b")

    # c: 逐 SNR 增益柱（展示效应量级）
    ax = fig.add_subplot(gs[0, 2])
    delta = [ps[str(s)]["delta"] * 100 for s in SNRS]
    bars = ax.bar(X, delta, 0.55, color=PAL["bayes"], edgecolor="none")
    for i, bb in enumerate(bars):
        ax.text(bb.get_x() + bb.get_width() / 2, delta[i], f"{delta[i]:.3f}",
                ha="center", va="bottom", fontsize=6, color=PAL["ink"])
    ax.set_xticks(X); ax.set_xticklabels(SNRS)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Absolute ΔBER (%)")
    ax.set_title("Effect size", loc="left")
    ax.set_ylim(0, max(delta) * 1.22)
    ax.grid(True, axis="y", alpha=0.6)
    panel_tag(ax, "c")

    save(fig, "fig3_mechanism")


def main():
    D = load()
    print("生成期刊级图……")
    fig_offline()
    fig_confirm(D)
    fig_mechanism(D)
    print("完成 →", OUT)


if __name__ == "__main__":
    main()

"""PPT 用单图（期刊风格）。输出到 figures/v2/，供 make_ppt.py 引用。"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "docs" / "results"
OUT = HERE / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

PAL = {
    "frozen": "#767676", "bayes": "#0F4D92",
    "t1": "#A8A8A8", "t2": "#4D4D4D", "t3": "#D0D0D0",
    "up": "#2E9E44", "ink": "#272727", "grid": "#E3E3E3",
    "teal": "#42949E", "violet": "#9A4D8E", "red": "#B64342",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.9, "axes.edgecolor": PAL["ink"],
    "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "legend.frameon": False, "lines.linewidth": 1.8,
    "grid.color": PAL["grid"], "grid.linewidth": 0.7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

SNRS = [0, 5, 10, 15]
X = np.arange(4)
LBL = [f"{s} dB" for s in SNRS]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("  ✓", name)


def load():
    d = {}
    d["conf"] = json.loads((RES / "bayesian_cir_confirmation_v1_20261106/summary.json").read_text(encoding="utf-8"))
    d["rows"] = json.loads((RES / "bayesian_cir_confirmation_v1_20261106/confirmation.json").read_text(encoding="utf-8"))["rows"]
    for k, f in [("5x32", "summary_5x32.json"), ("10x64", "summary_10x64.json"), ("10x128", "summary_10x128.json")]:
        d[k] = json.loads((RES / "bayesian_cir_diag_v1_20261106" / f).read_text(encoding="utf-8"))
    return d


def fig_offline():
    nn = [22.1947, 5.9599, 1.0598, 0.2156]
    dfe = [44.0524, 18.2986, 6.9090, 4.3743]
    lm = [44.0054, 22.4306, 7.2229, 5.0456]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.semilogy(X, dfe, "s--", color=PAL["t2"], ms=6, label="CFO+DD-Phase DFE-RLS")
    ax.semilogy(X, lm, "^:", color=PAL["t1"], ms=6, label="CFO+DD-Phase LMMSE-FIR")
    ax.semilogy(X, nn, "o-", color=PAL["bayes"], ms=7, label="Neural equalizer (ours)")
    for i in range(4):
        ax.annotate(f"−{(dfe[i]-nn[i])/dfe[i]*100:.0f}%", (X[i], nn[i]),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=9, color=PAL["up"], fontweight="bold")
    ax.set_xticks(X); ax.set_xticklabels(LBL)
    ax.set_xlabel("SNR"); ax.set_ylabel("Data BER")
    ax.grid(True, axis="y", which="both", alpha=0.6)
    ax.legend(loc="lower left", handlelength=2.0)
    ax.set_ylim(0.1, 90); ax.set_xlim(-0.4, 3.45)
    fig.tight_layout(); save(fig, "fig_offline_ber")


def fig_traditional():
    fair = {
        "Bayes": [0.181424, 0.049344, 0.007270, 0.001383],
        "Frozen": [0.214572, 0.052544, 0.008138, 0.001519],
        "DFE-RLS": [0.406359, 0.170546, 0.035862, 0.017442],
        "LMMSE-FIR": [0.436551, 0.149495, 0.041314, 0.016710],
        "SC-FDE": [0.490641, 0.485677, 0.484185, 0.482829],
    }
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    order = [("Bayes", PAL["bayes"], "o", "Bayes CIR"), ("Frozen", PAL["frozen"], "s", "Frozen"),
             ("DFE-RLS", "#4D4D4D", "^", "DFE-RLS"), ("LMMSE-FIR", "#A8A8A8", "v", "LMMSE-FIR"),
             ("SC-FDE", "#D0D0D0", "D", "SC-FDE-MMSE")]
    for name, c, mk, lab in order:
        ax.semilogy(X, fair[name], ls="none", marker=mk, color=c, ms=7, label=lab, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(8e-4, 1.6)
    ax.set_xticks(X); ax.set_xticklabels(LBL)
    ax.set_xlabel("SNR"); ax.set_ylabel("Data BER")
    ax.grid(True, axis="y", which="both", alpha=0.6)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(); save(fig, "fig_traditional_fair")


def fig_confirm(D):
    ps = D["conf"]["per_snr"]; rows = D["rows"]
    frozen = [np.mean([r["frozen_ber"] for r in rows if r["snr_db"] == s]) for s in SNRS]
    bayes = [np.mean([r["bayes_ber"] for r in rows if r["snr_db"] == s]) for s in SNRS]
    delta = [ps[str(s)]["delta"] for s in SNRS]
    ci = [1.96 * ps[str(s)]["se"] for s in SNRS]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.05, 1]})
    a1.semilogy(X, frozen, "o-", color=PAL["frozen"], ms=7, label="Frozen (acquisition CIR)")
    a1.semilogy(X, bayes, "s-", color=PAL["bayes"], ms=7, label="Bayes CIR (online)")
    a1.set_xticks(X); a1.set_xticklabels(LBL)
    a1.set_xlabel("SNR"); a1.set_ylabel("Data BER")
    a1.grid(True, axis="y", which="both", alpha=0.6)
    a1.legend(loc="lower left", handlelength=2.0)
    a1.set_title("a  Confirmed BER", loc="left")
    a1.set_xlim(-0.4, 3.4)

    y = np.arange(4)[::-1]
    a2.axvline(0, color=PAL["t2"], lw=0.9, ls=(0, (3, 2)))
    a2.errorbar(delta, y, xerr=ci, fmt="D", color=PAL["bayes"], ecolor=PAL["bayes"],
                elinewidth=1.4, capsize=4, ms=6)
    for i in range(4):
        a2.annotate(f"+{delta[i]:.4f}", (delta[i], y[i]), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=9.5)
    a2.set_yticks(y); a2.set_yticklabels(LBL)
    a2.set_xlabel("ΔBER = BER(Frozen) − BER(Bayes)")
    a2.grid(True, axis="x", alpha=0.6)
    a2.set_title("b  Paired gain, 95% CI", loc="left")
    a2.set_xlim(-0.002, max(delta) * 1.5)
    fig.tight_layout(w_pad=2.4); save(fig, "fig_ber_forest")


def fig_seed(D):
    ps = D["conf"]["per_snr"]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9))
    rng = np.random.default_rng(7)
    for ax, s in zip(axes, SNRS):
        v = np.array(ps[str(s)]["deltas"]) * 1e3
        m = ps[str(s)]["delta"] * 1e3
        ax.scatter(rng.uniform(-0.15, 0.15, 10), v, s=38, color=PAL["bayes"],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color=PAL["t2"], lw=0.8, ls=(0, (3, 2)))
        ax.plot([-0.28, 0.28], [m, m], color=PAL["ink"], lw=1.8, zorder=4)
        ax.set_xlim(-0.45, 0.45); ax.set_xticks([])
        ax.set_title(f"{s} dB", loc="left", fontsize=10)
        ax.grid(True, axis="y", alpha=0.55); ax.margins(y=0.22)
        ax.text(0.96, 0.95, f"+{ps[str(s)]['delta']:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
    axes[0].set_ylabel("Session ΔBER ×1000")
    fig.tight_layout(); save(fig, "fig_seed")


def fig_growth(D):
    h3 = D["conf"]["h3"]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    bins = np.arange(1, 9)
    for s, c in zip(SNRS, [PAL["bayes"], PAL["teal"], PAL["violet"], PAL["red"]]):
        yv = np.array(h3[str(s)]["bins"]) * 1e3
        ax.plot(bins, yv, "o-", color=c, ms=5, label=f"{s} dB")
        ax.text(8.15, yv[-1], f"{s} dB", color=c, fontsize=9, va="center", weight="bold")
    ax.set_yscale("log"); ax.set_xlim(0.85, 8.9)
    ax.set_xticks(bins)
    ax.set_xlabel("Frame window (16 frames each)")
    ax.set_ylabel("Window ΔBER ×1000")
    ax.grid(True, axis="y", which="both", alpha=0.5)
    fig.tight_layout(); save(fig, "fig_growth")


def fig_nmse(D):
    ps = D["conf"]["per_snr"]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    f = [ps[str(s)]["nmse_frozen"] for s in SNRS]
    b = [ps[str(s)]["nmse_bayes"] for s in SNRS]
    w = 0.34
    ax.bar(X - w / 2, f, w, color=PAL["frozen"], edgecolor="none", label="Frozen")
    ax.bar(X + w / 2, b, w, color=PAL["bayes"], edgecolor="none", label="Bayes")
    for i in range(4):
        ax.annotate(f"−{(f[i]-b[i])/f[i]*100:.0f}%", (X[i] + w / 2, b[i]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=PAL["up"], weight="bold")
    ax.set_xticks(X); ax.set_xticklabels(LBL)
    ax.set_ylabel("CIR NMSE")
    ax.set_ylim(0, max(f) * 1.3)
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(loc="upper right")
    fig.tight_layout(); save(fig, "fig_nmse")


def fig_scale(D):
    sc = [("5×32", D["5x32"]), ("10×64", D["10x64"]), ("10×128", D["10x128"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    xs = np.arange(3)
    for s, c in zip(SNRS, [PAL["bayes"], PAL["teal"], PAL["violet"], PAL["red"]]):
        g = np.array([d[str(s)]["gain"] for _, d in sc]) * 1e3
        ax.plot(xs, g, "o-", color=c, ms=5, label=f"{s} dB")
    ax.axhline(0, color=PAL["t2"], lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in sc])
    ax.set_xlabel("Development scale")
    ax.set_ylabel("ΔBER ×1000")
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(ncol=2, loc="center left", fontsize=9)
    fig.tight_layout(); save(fig, "fig_scale")


def main():
    D = load()
    fig_offline(); fig_traditional(); fig_confirm(D)
    fig_seed(D); fig_growth(D); fig_nmse(D); fig_scale(D)
    print("完成 →", OUT)


if __name__ == "__main__":
    main()

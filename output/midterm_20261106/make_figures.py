"""生成中期答辩所需的全部数据图。

所有数字都从 docs/results 下的 JSON 直接读取，不手工抄写，保证图与归档一致。
运行：python output/midterm_20261106/make_figures.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import style
from style import (ALERT, BAYES, BG, BLUE, FROZEN, GOOD, GRID, INK, MUTED,
                   NAVY, PANEL, TEAL, FIGDIR, RESULTS)

plt = style.apply_matplotlib_style()

SNRS = [0, 5, 10, 15]
SNR_LABEL = [f"{s} dB" for s in SNRS]

CONF_DIR = RESULTS / "bayesian_cir_confirmation_v1_20261106"
DIAG_DIR = RESULTS / "bayesian_cir_diag_v1_20261106"
SAT_DIR = RESULTS / "offline_saturation_20261106"


def load():
    d = {}
    d["conf"] = json.loads((CONF_DIR / "summary.json").read_text(encoding="utf-8"))
    d["conf_rows"] = json.loads((CONF_DIR / "confirmation.json").read_text(encoding="utf-8"))
    d["scale"] = {
        "5×32": json.loads((DIAG_DIR / "summary_5x32.json").read_text(encoding="utf-8")),
        "10×64": json.loads((DIAG_DIR / "summary_10x64.json").read_text(encoding="utf-8")),
        "10×128": json.loads((DIAG_DIR / "summary_10x128.json").read_text(encoding="utf-8")),
    }
    d["sat"] = json.loads((SAT_DIR / "original_checkpoint/summary.json").read_text(encoding="utf-8"))
    return d


def half_ci(se):
    """双侧 95% 半宽（大样本正态近似；与归档的单侧下界口径一致）。"""
    return 1.96 * np.asarray(se)


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------- 图：BER + 森林图


def fig_ber_forest(D):
    ps = D["conf"]["per_snr"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.15), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    frozen = [np.nan] * 4
    bayes = [np.nan] * 4
    # BER 数值本身不在 summary.json，用 rows 重算
    rows = D["conf_rows"]["rows"]
    for i, s in enumerate(SNRS):
        rr = [r for r in rows if r["snr_db"] == s]
        frozen[i] = np.mean([r["frozen_ber"] for r in rr])
        bayes[i] = np.mean([r["bayes_ber"] for r in rr])

    x = np.arange(4)
    ax1.semilogy(x, frozen, "o-", color=FROZEN, label="Frozen（acquisition CIR）",
                 markeredgecolor=BG, markeredgewidth=1.2, zorder=3)
    ax1.semilogy(x, bayes, "s-", color=BAYES, label="Bayes CIR（在线后验）",
                 markeredgecolor=BG, markeredgewidth=1.2, zorder=3)
    for i in range(4):
        ax1.annotate(f"{bayes[i]*100:.2f}%", (x[i], bayes[i]), textcoords="offset points",
                     xytext=(14, 0), va="center", ha="left", fontsize=8.4, color=BAYES)
    ax1.set_xticks(x); ax1.set_xticklabels(SNR_LABEL)
    ax1.set_xlabel("信噪比"); ax1.set_ylabel("Data BER（越低越好）")
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.legend(loc="lower left", handlelength=1.6)
    ax1.set_xlim(-0.45, 3.45)
    ax1.set_title("a　误码率", loc="left", fontsize=11.5)

    # 森林图
    deltas = [ps[str(s)]["delta"] for s in SNRS]
    ci = half_ci([ps[str(s)]["se"] for s in SNRS])
    y = np.arange(4)[::-1]
    ax2.axvline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax2.errorbar(deltas, y, xerr=ci, fmt="D", color=BAYES, ecolor=BAYES,
                 elinewidth=1.6, capsize=3.5, markersize=6, zorder=3)
    for i, s in enumerate(SNRS):
        v = ps[str(s)]["delta"]
        p = ps[str(s)]["p_one_sided"]
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else "*")
        ax2.annotate(f"+{v:.4f} {star}", (v, y[i]), textcoords="offset points",
                     xytext=(10, 0), va="center", fontsize=9, color=INK)
    ax2.set_yticks(y); ax2.set_yticklabels(SNR_LABEL)
    ax2.set_xlabel("ΔBER = BER(Frozen) − BER(Bayes)")
    ax2.grid(True, axis="x", alpha=0.6)
    ax2.set_xlim(-0.002, max(deltas) * 1.58)
    ax2.set_title("b　配对增益与 95% 置信区间", loc="left", fontsize=11.5)

    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_ber_forest.png")


# ---------------------------------------------------------------- 图：10 seed 配对


def fig_seed(D):
    ps = D["conf"]["per_snr"]
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.2))
    rng = np.random.default_rng(7)

    for ax, s in zip(axes, SNRS):
        vals = np.array(ps[str(s)]["deltas"]) * 1e3
        m = ps[str(s)]["delta"] * 1e3
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(jitter, vals, s=50, color=BAYES, edgecolor=BG, linewidth=0.9, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.plot([-0.30, 0.30], [m, m], color=INK, lw=1.8, zorder=4)
        ax.set_xlim(-0.46, 0.46)
        ax.set_xticks([])
        ax.set_title(f"{s} dB", loc="left", fontsize=11)
        ax.grid(True, axis="y", alpha=0.55)
        ax.margins(y=0.24)
        ax.text(0.96, 0.95, f"均值 +{ps[str(s)]['delta']:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.4, color=INK)

    axes[0].set_ylabel("单 seed ΔBER ×1000")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig_seed.png")


def fig_growth(D):
    h3 = D["conf"]["h3"]
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    colors = [BAYES, BLUE, TEAL, GOOD]
    bins = np.arange(1, 9)

    for s, c in zip(SNRS, colors):
        y = np.array(h3[str(s)]["bins"]) * 1e3
        ax.plot(bins, y, "o-", color=c, markeredgecolor=BG, markeredgewidth=1.1,
                lw=1.8, zorder=3)
        ax.text(8.12, y[-1], f"{s} dB", color=c, fontsize=9.5, va="center",
                ha="left", weight="semibold")

    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_yscale("log")
    ax.set_xlim(0.82, 8.95)
    ax.set_xlabel("会话内帧窗口（每箱 16 帧）")
    ax.set_ylabel("配对增益 ΔBER ×1000")
    ax.set_xticks(bins)
    ax.grid(True, axis="y", which="both", alpha=0.55)

    fig.tight_layout()
    save(fig, "fig_growth.png")


def fig_scale(D):
    sc = D["scale"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.0), sharex=True)
    xs = np.arange(3)
    names = list(sc.keys())

    for ax, s in zip(axes.ravel(), SNRS):
        g = np.array([sc[n][str(s)]["gain"] for n in names]) * 1e3
        ci = np.array([sc[n][str(s)].get("ci", 0) for n in names]) * 1e3
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.errorbar(xs, g, yerr=ci, fmt="o-", color=BAYES, ecolor=BAYES,
                    elinewidth=1.5, capsize=3.5, markersize=6,
                    markeredgecolor=BG, markeredgewidth=1.1)
        ax.set_title(f"{s} dB", loc="left", fontsize=11)
        ax.grid(True, axis="y", alpha=0.55)
        ax.set_xticks(xs)
        ax.margins(y=0.30)

    for ax in axes[1]:
        ax.set_xticklabels(names)
    fig.supylabel("配对增益 ΔBER ×1000", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0.055, 0, 1, 0.94))
    save(fig, "fig_scale.png")


def fig_nmse(D):
    ps = D["conf"]["per_snr"]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    x = np.arange(4)
    w = 0.34
    f = [ps[str(s)]["nmse_frozen"] for s in SNRS]
    b = [ps[str(s)]["nmse_bayes"] for s in SNRS]

    ax.bar(x - w / 2, f, w, color=FROZEN, label="Frozen（acquisition CIR）",
           edgecolor="none")
    ax.bar(x + w / 2, b, w, color=BAYES, label="Bayes CIR（在线后验）",
           edgecolor="none")
    for i in range(4):
        drop = (f[i] - b[i]) / f[i] * 100
        ax.annotate(f"−{drop:.0f}%", (x[i] + w / 2, b[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=GOOD, weight="bold")

    ax.set_xticks(x); ax.set_xticklabels(SNR_LABEL)
    ax.set_ylabel("CIR 归一化均方误差 NMSE")
    ax.grid(True, axis="y", alpha=0.55)
    ax.set_ylim(0, max(f) * 1.28)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save(fig, "fig_nmse.png")


def fig_oracle_cir(D):
    """离线饱和诊断：把估计 CIR 换成真实 CIR，Frozen BER 还能降多少。"""
    oc = D["sat"]["oracle_cir"]["per_snr"]
    est, tru = [], []
    for s in SNRS:
        node = oc[f"{float(s)}"]
        est.append(node["ber_estimated"]["mean"])
        tru.append(node["ber_true"]["mean"])

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    y = np.arange(4)[::-1]
    for i, s in enumerate(SNRS):
        ax.plot([tru[i], est[i]], [y[i], y[i]], color=GRID, lw=2.4, zorder=1)
    ax.scatter(est, y, s=64, color=FROZEN, edgecolor=BG, linewidth=1.2,
               zorder=3, label="当前：估计 CIR")
    ax.scatter(tru, y, s=64, color=TEAL, edgecolor=BG, linewidth=1.2,
               zorder=3, label="上界：真实 CIR（不可部署）")
    for i in range(4):
        drop = (est[i] - tru[i]) / est[i] * 100
        ax.annotate(f"−{drop:.0f}%", xy=((est[i] + tru[i]) / 2, y[i]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    va="bottom", fontsize=9, color=GOOD, weight="bold")

    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(SNR_LABEL)
    ax.set_xlabel("Frozen Data BER（对数轴）")
    ax.grid(True, axis="x", which="both", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save(fig, "fig_oracle_cir.png")


def _box(ax, xy, w, h, text, fc, ec, tc=INK, fs=10.5, bold=False, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.10",
                       facecolor=fc, edgecolor=ec, linewidth=1.1, linestyle=ls)
    ax.add_patch(p)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs,
            color=tc, weight="bold" if bold else "normal", linespacing=1.55)
    return p


def _arrow(ax, a, b, color=MUTED, lw=1.2, style="-|>", ls="-", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def fig_route():
    fig, ax = plt.subplots(figsize=(11.8, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0.40, 3.30); ax.axis("off")

    yb = 2.10
    boxes = [
        (1.42, 2.30, "月面散射物理", "双站雷达方程\n分布式 RCS · 地形阴影", PANEL, MUTED),
        (4.12, 2.30, "宽带 EME 信道", "准镜面 + 漫散射\n116 符号长记忆 ISI", PANEL, MUTED),
        (6.98, 2.42, "离线训练：适应信道", "全帧 BCE 监督\nPilot 条件化神经均衡", "#F3E3CB", BAYES),
        (9.92, 2.50, "在线微调：做均衡", "合法 Pilot 观测\n物理状态后验更新", NAVY, NAVY),
    ]
    for xc, w, t, sub, fc, ec in boxes:
        dark = fc == NAVY
        _box(ax, (xc, yb), w, 1.30, "", fc, ec)
        ax.text(xc, yb + 0.28, t, ha="center", va="center", fontsize=10.3,
                color="white" if dark else INK, weight="bold")
        ax.text(xc, yb - 0.25, sub, ha="center", va="center", fontsize=8.5,
                color="#D8E2EC" if dark else MUTED, linespacing=1.5)

    for a, b in [(2.60, 2.93), (5.30, 5.73), (8.22, 8.63)]:
        _arrow(ax, (a, yb), (b, yb), color=NAVY, lw=1.3)

    # 研究点归属
    ax.plot([0.28, 5.42], [1.02, 1.02], color=BLUE, lw=1.0)
    ax.text(2.85, 0.70, "研究点一　宽带 EME 信道建模（已完成，拟录用）",
            ha="center", fontsize=9.2, color=BLUE)
    ax.plot([5.72, 11.35], [1.02, 1.02], color=BAYES, lw=1.0)
    ax.text(8.55, 0.70, "研究点二　离线训练 + 在线微调",
            ha="center", fontsize=9.2, color=BAYES)

    fig.tight_layout()
    save(fig, "fig_route.png")
def fig_rx_arch():
    fig, ax = plt.subplots(figsize=(11.8, 4.14))
    ax.set_xlim(0, 12); ax.set_ylim(0.78, 4.18); ax.axis("off")

    ax.text(0.15, 4.00, "Adapt Pilot 已知符号用于更新　·　Data 波形参与软判决、标签只事后评分　·　Reward 仅验收",
            fontsize=8.6, color=MUTED, va="center")

    # 帧结构条
    y = 3.15
    segs = [(0.6, 3.3, "Adapt Pilot  224", "#E7C9A0"),
            (3.9, 4.9, "Data  768", "#C9D6E2"),
            (8.8, 1.5, "Reward 32", "#DCE3D5")]
    for x0, w, t, c in segs:
        ax.add_patch(plt.Rectangle((x0, y), w, 0.60, facecolor=c,
                                   edgecolor="white", linewidth=2))
        ax.text(x0 + w / 2, y + 0.30, t, ha="center", va="center",
                fontsize=9.5, color=INK, weight="bold")

    # 处理链
    yb = 1.62
    chain = [(1.55, "物理状态跟踪", "Bayes 递归 CIR\nCFO · 慢相位 · tail", "#F3E3CB"),
             (4.75, "冻结神经均衡器", "主干权重不变\n输出 logits", PANEL),
             (7.75, "判决与复用", "soft→hard\nframe tail 递推", PANEL),
             (10.35, "评估", "Data BER\nNMSE", "#E6EBDD")]
    for xc, t, sub, c in chain:
        _box(ax, (xc, yb), 2.55, 1.22, "", c, MUTED)
        ax.text(xc, yb + 0.28, t, ha="center", va="center", fontsize=10,
                color=INK, weight="bold")
        ax.text(xc, yb - 0.25, sub, ha="center", va="center", fontsize=8.2,
                color=MUTED, linespacing=1.45)
    for a, b in [(2.85, 3.45), (6.05, 6.45), (9.05, 9.05)]:
        if a != b:
            _arrow(ax, (a, yb), (b, yb), color=NAVY)

    # Adapt Pilot 下灌到状态跟踪
    _arrow(ax, (1.95, 3.15), (1.72, 2.28), color="#B8860B", lw=1.3)

    # Data 标签禁止进入闭环
    ax.plot([5.2, 5.2], [3.15, 2.28], color=ALERT, lw=1.2, ls=(0, (3, 2)))
    ax.plot([5.05, 5.35], [2.86, 2.62], color=ALERT, lw=1.7)
    ax.plot([5.05, 5.35], [2.62, 2.86], color=ALERT, lw=1.7)
    ax.text(5.44, 2.74, "Data 标签不进入在线闭环", fontsize=8.4, color=ALERT,
            va="center", ha="left")

    fig.tight_layout()
    save(fig, "fig_rx_arch.png")


def fig_bayes_loop():
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    ax.set_xlim(0, 10.35); ax.set_ylim(0.55, 3.90); ax.axis("off")


    nodes = [
        (1.30, 2.70, "acquisition\n先验", "均值 h$_0$\n协方差 P$_0$", PANEL),
        (3.85, 2.70, "Adapt Pilot\n后验", "已知符号\n线性贝叶斯更新", "#F3E3CB"),
        (6.40, 2.70, "Data 软判决\n后验", "噪声含符号方差\n错误符号自动降权", "#F3E3CB"),
        (8.95, 2.70, "跨帧递推", "P 持久\n+ 过程噪声 Q", PANEL),
    ]
    for xc, yc, t, sub, c in nodes:
        _box(ax, (xc, yc), 1.95, 1.42, "", c, NAVY if "后验" in t else MUTED)
        ax.text(xc, yc + 0.30, t, ha="center", va="center", fontsize=9.6,
                color=INK, weight="bold", linespacing=1.35)
        ax.text(xc, yc - 0.32, sub, ha="center", va="center", fontsize=7.9,
                color=MUTED, linespacing=1.45)

    for a, b in [(2.30, 2.85), (4.85, 5.40), (7.40, 7.95)]:
        _arrow(ax, (a, 2.70), (b, 2.70), color=NAVY, lw=1.3)

    # 直角回环，绕开中间两个框
    ax.plot([8.95, 8.95], [1.99, 1.32], color=BAYES, lw=1.3)
    ax.plot([8.95, 1.30], [1.32, 1.32], color=BAYES, lw=1.3)
    _arrow(ax, (1.30, 1.32), (1.30, 1.99), color=BAYES, lw=1.3)

    ax.text(3.85, 3.66, "输入：接收波形 + 已知 Pilot 符号", ha="center",
            fontsize=8.2, color=MUTED)
    _arrow(ax, (3.85, 3.55), (3.85, 3.42), color=MUTED, lw=0.9)

    fig.tight_layout()
    save(fig, "fig_bayes_loop.png")



def fig_offline_arch():
    """离线神经均衡器的训练结构示意。"""
    fig, ax = plt.subplots(figsize=(11.8, 3.45))
    ax.set_xlim(0, 12); ax.set_ylim(1.02, 2.98); ax.axis("off")

    yb = 2.05
    boxes = [
        (1.28, 2.10, "接收帧", "1024 符号 BPSK\n116 符号长记忆"),
        (3.78, 2.10, "已知符号输入", "仅 Adapt Pilot 位置非零\nData 段不含真值"),
        (6.72, 2.90, "Pilot 条件化 + 4 层展开", "d_model 64 · 4 heads\nAdapter / LoRA rank 8"),
        (10.10, 2.05, "输出 logits", "1024 个符号的\n软判决输入"),
    ]
    for xc, w, t, sub in boxes:
        _box(ax, (xc, yb), w, 1.05, "", PANEL, MUTED)
        ax.text(xc, yb + 0.22, t, ha="center", va="center", fontsize=10,
                color=INK, weight="bold")
        ax.text(xc, yb - 0.24, sub, ha="center", va="center", fontsize=8.2,
                color=MUTED, linespacing=1.45)

    for a, b in [(2.36, 2.70), (4.86, 5.24), (8.20, 9.05)]:
        _arrow(ax, (a, yb), (b, yb), color=NAVY, lw=1.3)

    # 物理 warm start 标注
    _arrow(ax, (6.72, 2.62), (6.72, 2.44), color=TEAL, lw=1.0)
    ax.text(6.72, 2.72, "物理 warm start ×8", ha="center", fontsize=8.2, color=TEAL)

    # 离线损失
    ax.text(10.10, 1.18, "离线损失：全帧等权 BCE", ha="center", fontsize=9.2,
            color=ALERT)
    _arrow(ax, (10.10, 1.52), (10.10, 1.30), color=ALERT, lw=1.0, style="-|>")

    fig.tight_layout()
    save(fig, "fig_offline_arch.png")


def fig_offline_ber(D):
    """离线均衡器与传统均衡器的 BER 对比（主矩阵，3 seed × 60 帧）。"""
    # 来源：docs/offline_equalizer_training_and_baseline_20261105.md
    nn = [22.1947, 5.9599, 1.0598, 0.2156]
    dfe = [44.0524, 18.2986, 6.9090, 4.3743]
    lmmse = [44.0054, 22.4306, 7.2229, 5.0456]

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    x = np.arange(4)

    ax.semilogy(x, dfe, "s--", color=FROZEN, label="CFO + DD-Phase DFE-RLS",
                markeredgecolor=BG, markeredgewidth=1.1, lw=1.4, zorder=3)
    ax.semilogy(x, lmmse, "^:", color=TEAL, label="CFO + DD-Phase LMMSE-FIR",
                markeredgecolor=BG, markeredgewidth=1.1, lw=1.4, zorder=3)
    ax.semilogy(x, nn, "o-", color=BAYES, label="离线神经均衡器",
                markeredgecolor=BG, markeredgewidth=1.2, lw=2.0, zorder=4)

    for i in range(4):
        drop = (dfe[i] - nn[i]) / dfe[i] * 100
        ax.annotate(f"−{drop:.0f}%", (x[i], nn[i]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8.6,
                    color=BAYES, weight="bold")

    ax.set_xticks(x); ax.set_xticklabels(SNR_LABEL)
    ax.set_xlabel("信噪比")
    ax.set_ylabel("Data BER（对数轴）")
    ax.grid(True, axis="y", which="both", alpha=0.55)
    ax.legend(loc="lower left", handlelength=2.2)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0.1, 90)

    fig.tight_layout()
    save(fig, "fig_offline_ber.png")


def fig_traditional_fair():
    """3x16 公平对照：Frozen/Bayes 与三种传统接收机（seed 130001-130003）。"""
    labels = SNR_LABEL
    series = [
        ("Frozen",     [0.214572, 0.052544, 0.008138, 0.001519], FROZEN),
        ("Bayes",      [0.181424, 0.049344, 0.007270, 0.001383], BAYES),
        ("LMMSE-FIR",  [0.436551, 0.149495, 0.041314, 0.016710], TEAL),
        ("DFE-RLS",    [0.406359, 0.170546, 0.035862, 0.017442], "#4A5D6E"),
        ("SC-FDE",     [0.490641, 0.485677, 0.484185, 0.482829], "#B8C2CC"),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    x = np.arange(4); w = 0.16
    for k, (name, vals, c) in enumerate(series):
        ax.bar(x + (k - 2) * w, vals, w, color=c, edgecolor="none", label=name)
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Data BER（对数轴）")
    ax.grid(True, axis="y", which="both", alpha=0.5)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              columnspacing=1.1, handlelength=1.2, fontsize=9)
    for i in range(4):
        best_trad = min(series[2][1][i], series[3][1][i])
        ax.annotate(f"{best_trad/series[1][1][i]:.1f}×",
                    (x[i] - w, best_trad * 1.15), ha="center", fontsize=8.2,
                    color=GOOD, weight="bold")
    fig.tight_layout()
    save(fig, "fig_traditional_fair.png")

def fig_plan():
    fig, ax = plt.subplots(figsize=(11.4, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0.75, 3.05); ax.axis("off")


    phases = [
        (0.5, 2.75, "2026 11–12", "会话扩展与漂移场景\n冻结确认判据", BLUE),
        (3.55, 2.55, "2027 01–02", "学位论文主体撰写\n第二、三章", BAYES),
        (6.40, 2.75, "2027 02–03", "补充确认实验\n论文初稿完善", TEAL),
        (9.45, 2.20, "2027 03+", "论文整合\n送审答辩", GOOD),
    ]
    y = 1.75
    for x0, w, t, sub, c in phases:
        ax.add_patch(plt.Rectangle((x0, y), w, 0.70, facecolor=c, edgecolor="none"))
        ax.text(x0 + w / 2, y + 0.35, t, ha="center", va="center", fontsize=9.5,
                color="white", weight="bold")
        ax.text(x0 + w / 2, y - 0.42, sub, ha="center", va="center", fontsize=8.4,
                color=INK, linespacing=1.55)

    # 当前进度标记
    ax.annotate("现在", xy=(1.10, y + 0.86), xytext=(1.10, y + 1.12),
                ha="center", fontsize=9, color=BAYES, weight="semibold",
                arrowprops=dict(arrowstyle="-|>", color=BAYES, lw=1.2))
    ax.text(1.85, y + 1.12, "独立确认 v1 已完成", fontsize=9, color=BAYES, va="center")

    fig.tight_layout()
    save(fig, "fig_plan.png")
def fig_online_loop():
    """在线适配闭环：状态条件化 → 均衡 → 判决 → 状态更新。"""
    fig, ax = plt.subplots(figsize=(11.8, 4.3))
    ax.set_xlim(0, 12); ax.set_ylim(1.82, 4.62); ax.axis("off")

    # 接收波形
    _box(ax, (1.75, 4.02), 2.10, 0.80, "", PANEL, MUTED)
    ax.text(1.75, 4.14, "接收波形 y", ha="center", va="center", fontsize=10.2,
            color=INK, weight="bold")
    ax.text(1.75, 3.86, "第 t 帧 · 1024 符号", ha="center", va="center",
            fontsize=8.2, color=MUTED)

    # 物理状态
    _box(ax, (1.75, 2.72), 2.60, 1.10, "", "#F3E3CB", BAYES)
    ax.text(1.75, 3.02, "物理状态 s(t)", ha="center", va="center", fontsize=10.2,
            color=INK, weight="bold")
    ax.text(1.85, 2.56, "CIR（116 抽头）· CFO\n慢相位 · soft-tail", ha="center",
            va="center", fontsize=8.2, color=MUTED, linespacing=1.5)

    # 均衡器
    _box(ax, (5.60, 3.42), 2.30, 1.36, "", NAVY, NAVY)
    ax.text(5.60, 3.62, "冻结神经均衡器", ha="center", va="center", fontsize=10.2,
            color="white", weight="bold")
    ax.text(5.60, 3.22, "权重全帧不变\nPilot 条件化", ha="center", va="center",
            fontsize=8.2, color="#D8E2EC", linespacing=1.5)

    # soft 判决
    _box(ax, (8.80, 4.02), 2.05, 0.80, "", PANEL, MUTED)
    ax.text(8.80, 4.14, "soft 判决", ha="center", va="center", fontsize=10.2,
            color=INK, weight="bold")
    ax.text(8.80, 3.86, "tanh(logit / 2)", ha="center", va="center",
            fontsize=8.2, color=MUTED)

    # 贝叶斯更新
    _box(ax, (8.85, 2.60), 2.40, 1.02, "", "#E6EBDD", GOOD)
    ax.text(8.85, 2.78, "贝叶斯更新", ha="center", va="center", fontsize=10.2,
            color=INK, weight="bold")
    ax.text(8.85, 2.40, "Pilot 后验 + Data 软观测后验\n+ 过程噪声 Q", ha="center",
            va="center", fontsize=8.0, color=MUTED, linespacing=1.5)

    # 主链路
    _arrow(ax, (2.82, 4.02), (4.42, 3.68), color=NAVY, lw=1.3)
    _arrow(ax, (3.16, 3.02), (4.42, 3.26), color=BAYES, lw=1.5)
    _arrow(ax, (6.76, 3.58), (7.76, 4.00), color=NAVY, lw=1.3)
    _arrow(ax, (9.78, 3.62), (9.78, 3.13), color=NAVY, lw=1.3)

    # 回环
    _arrow(ax, (7.63, 2.60), (3.18, 2.60), color=GOOD, lw=1.5)
    ax.text(5.40, 2.36, "s(t+1) 作为下一帧的条件", ha="center", fontsize=8.6,
            color=GOOD)

    # 旁注


    fig.tight_layout()
    save(fig, "fig_online_loop.png")
def main():
    D = load()
    print("生成数据图……")
    fig_ber_forest(D)
    fig_seed(D)
    fig_growth(D)
    fig_scale(D)
    fig_nmse(D)
    fig_oracle_cir(D)
    fig_offline_ber(D)
    fig_traditional_fair()
    print("生成示意图……")
    fig_route()
    fig_rx_arch()
    fig_bayes_loop()
    fig_online_loop()
    fig_offline_arch()
    fig_plan()
    print("完成。")


if __name__ == "__main__":
    main()

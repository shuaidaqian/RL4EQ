# -*- coding: utf-8 -*-
"""
对比可视化 — 4种均衡方法在同一信道下的 BER vs SNR 曲线

方法:
  1. MMSE 频域均衡器 (baseline)
  2. 纯因果预训练 (从头因果, V8 方案)
  3. 非因果预训练 + 在线因果微调 (无信道编码器)
  4. 非因果预训练 + 信道编码器 + 在线因果微调 (完整方案)

用法:
  python compare_methods.py
  python compare_methods.py --snr_range 0 25 --num_frames 100
"""

import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig, FrameGenerator
from env.comm_env import CommunicationEnv, EnvConfig
from env.rayleigh_channel import RayleighMultipathChannel
from agent.actor_critic_v2 import ActorCritic, TransformerConfig
from agent.channel_encoder import ChannelEncoder
from baseline.mmse_equalizer import MMSEEqualizer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_pretrained(pretrain_dir, device):
    """加载预训练模型."""
    cfg = TransformerConfig(
        state_dim=45, d_model=128, n_heads=4, n_layers=3, dim_feedforward=256,
    )
    ac = ActorCritic(cfg).to(device)
    chan_enc = ChannelEncoder(train_len=128, cond_dim=cfg.cond_dim).to(device)
    ckpt = torch.load(
        os.path.join(pretrain_dir, "model_best.pt"),
        map_location=device, weights_only=True,
    )
    ac.load_state_dict(ckpt["ac"])
    chan_enc.load_state_dict(ckpt["chan_enc"])
    ac.eval(); chan_enc.eval()
    return ac, chan_enc


def load_finetuned(finetune_dir, device):
    """加载微调后的模型."""
    cfg = TransformerConfig(
        state_dim=45, d_model=128, n_heads=4, n_layers=3, dim_feedforward=256,
    )
    ac = ActorCritic(cfg).to(device)
    chan_enc = ChannelEncoder(train_len=128, cond_dim=cfg.cond_dim).to(device)
    ckpt = torch.load(
        os.path.join(finetune_dir, "model_best.pt"),
        map_location=device, weights_only=True,
    )
    ac.load_state_dict(ckpt["ac"])
    chan_enc.load_state_dict(ckpt["chan_enc"])
    ac.eval(); chan_enc.eval()
    return ac, chan_enc


@torch.no_grad()
def eval_method1_mmse(env, snr_db):
    """MMSE 频域均衡."""
    eq = MMSEEqualizer(num_taps=16)
    rx = env.get_rx_symbols().numpy()
    tx_tr = env._bits[:128].numpy().astype(np.float32)
    rx_tr = rx[:128]
    s_est, _ = eq(
        torch.from_numpy(rx),
        torch.from_numpy(tx_tr),
        torch.from_numpy(rx_tr),
        snr_db,
    )
    preds = (torch.sigmoid(torch.from_numpy(s_est).float()) > 0.5).float()
    # MMSE 输出是软值, 用符号判决
    preds = (torch.from_numpy(s_est) > 0).float()
    true_bits = env.get_true_bits()
    ber = (preds != true_bits).float().mean().item()
    return ber


@torch.no_grad()
def eval_method2_causal_pretrain(ac, env, device):
    """纯因果预训练 (没有信道编码器)."""
    states = env.get_all_states().unsqueeze(0).to(device)
    true_bits = env.get_true_bits().unsqueeze(0).to(device)
    L = states.shape[1]
    mask = torch.triu(
        torch.ones(L, L, device=device) * float("-inf"), diagonal=1
    )
    probs, _, _ = ac.forward(states, mask=mask)
    preds = (probs.squeeze(-1) > 0.5).float()
    ber = (preds != true_bits).float().mean().item()
    return ber


@torch.no_grad()
def eval_method3_noncausal_finetune(ac, chan_enc, env, device):
    """非因果预训练 + 信道编码器 + 因果推理."""
    states = env.get_all_states().unsqueeze(0).to(device)
    true_bits = env.get_true_bits().unsqueeze(0).to(device)
    rx_all = env.get_rx_symbols().to(device)
    rx_train = rx_all[:128].unsqueeze(0)
    known_bits = env._bits[:128].to(device).unsqueeze(0)
    known_bits_sym = (1 - 2 * known_bits)

    cond = chan_enc(rx_train, known_bits_sym)
    L = states.shape[1]
    mask = torch.triu(
        torch.ones(L, L, device=device) * float("-inf"), diagonal=1
    )
    probs, _, _ = ac.forward(states, mask=mask, cond=cond)
    preds = (probs.squeeze(-1) > 0.5).float()
    ber = (preds != true_bits).float().mean().item()
    return ber


def run_one_frame(env, seed, snr_db, channel_cfg, device):
    """生成一帧测试数据."""
    base_cfg = EnvConfig(
        frame=FrameConfig(),
        channel=dict(**channel_cfg, snr_db=snr_db, seed=seed),
        window_K=10, seed=seed,
    )
    env = CommunicationEnv(base_cfg)
    env.reset()
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_min", type=float, default=0)
    parser.add_argument("--snr_max", type=float, default=25)
    parser.add_argument("--snr_step", type=float, default=2.5)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrain_dir", type=str, default="pretrained_nc")
    parser.add_argument("--finetune_dir", type=str, default="finetuned")
    args = parser.parse_args()

    device = torch.device(args.device)
    snr_range = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)

    # 加载模型
    print("加载模型...")
    ac_pretrain, ce_pretrain = load_pretrained(args.pretrain_dir, device)
    ac_finetune, ce_finetune = load_finetuned(args.finetune_dir, device)

    # 也可以用因果预训练 (从 pretrained/ 加载, 它用 actor_critic 旧版)
    # 这里简化: 用 pretrained/actor_critic_final.pt 加载 actor_critic_v2 能加载吗?
    # 不能, 因为结构不同. 所以我们用非因果预训练模型做因果推理作为"纯因果"对比
    # 实际上我们比较的是:
    # 1. MMSE
    # 2. 非因果预训练 + 因果推理 (无 cond)
    # 3. 非因果预训练 + 因果推理 + cond (有信道编码器)
    # 4. 微调后的模型

    results = {
        "MMSE": [],
        "Pretrain+Causal": [],
        "Pretrain+Causal+Cond": [],
        "Finetuned": [],
    }

    channel_cfg = dict(num_taps=16, delay_spread=10)

    for snr in snr_range:
        bers = {k: [] for k in results}
        for frame_idx in range(args.num_frames):
            seed = 42 + frame_idx
            env = run_one_frame(None, seed, snr, channel_cfg, device)

            # MMSE
            ber1 = eval_method1_mmse(env, snr)
            bers["MMSE"].append(ber1)

            # 非因果预训练 + 因果推理 (无 cond, 使用 cond_net 自提取)
            ber2 = eval_method2_causal_pretrain(ac_pretrain, env, device)
            bers["Pretrain+Causal"].append(ber2)

            # 非因果预训练 + 因果推理 + cond
            ber3 = eval_method3_noncausal_finetune(ac_pretrain, ce_pretrain, env, device)
            bers["Pretrain+Causal+Cond"].append(ber3)

            # 微调后模型
            ber4 = eval_method3_noncausal_finetune(ac_finetune, ce_finetune, env, device)
            bers["Finetuned"].append(ber4)

        for k in results:
            results[k].append(np.mean(bers[k]))
        print(f"  SNR={snr:4.1f}dB | MMSE={results['MMSE'][-1]:.5f} "
              f"Pretrain+Causal={results['Pretrain+Causal'][-1]:.5f} "
              f"+Cond={results['Pretrain+Causal+Cond'][-1]:.5f} "
              f"Finetuned={results['Finetuned'][-1]:.5f}")

    # 绘图
    if not HAS_MPL:
        print("matplotlib 不可用, 跳过绘图")
        return

    colors = {
        "MMSE": "#E53935",
        "Pretrain+Causal": "#1E88E5",
        "Pretrain+Causal+Cond": "#43A047",
        "Finetuned": "#FB8C00",
    }
    markers = {
        "MMSE": "s",
        "Pretrain+Causal": "o",
        "Pretrain+Causal+Cond": "^",
        "Finetuned": "D",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label in ["MMSE", "Pretrain+Causal", "Pretrain+Causal+Cond", "Finetuned"]:
        ax.plot(snr_range, results[label], color=colors[label],
                marker=markers[label], markersize=6, lw=2, label=label)

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 0.6)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("均衡方法对比: BER vs SNR (16抽头瑞利信道)", fontsize=13)

    # 添加参考线
    ax.axhline(0.01, color="gray", ls="--", alpha=0.4, label="1% target")
    ax.axhline(0.001, color="gray", ls=":", alpha=0.4, label="0.1% target")

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 表格输出
    print(f"\n{'=' * 75}")
    print(f"{'SNR(dB)':>8}", end="")
    for label in results:
        print(f"{label:>22}", end="")
    print()
    print(f"{'-' * 75}")
    for i, snr in enumerate(snr_range):
        print(f"{snr:>8.1f}", end="")
        for label in results:
            print(f"{results[label][i]:>22.5f}", end="")
        print()
    print(f"{'=' * 75}")
    print("图表已保存到 logs/method_comparison.png")


if __name__ == "__main__":
    main()

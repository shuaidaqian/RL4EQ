# -*- coding: utf-8 -*-
"""
在线微调 — 加载非因果预训练模型 + 在线因果掩码推理

核心思路:
  1. 加载预训练的 ActorCritic + ChannelEncoder
  2. 每帧先用训练序列提取 cond (信道特征)
  3. 在线推理时使用因果掩码 (只能看过去), cond 提供全局信道信息
  4. PPO 微调, 只在已知位置 (训练+导频) 给奖励

改进点 (相比 v1):
  - 修正动作值计算 (detach 避免二次 backward)
  - 奖励改为 ±1 更激进
  - 冻结策略: 前 50 帧冻结 channel_encoder + transformer 主体
  - 数据段给惩罚项 (-0.1) 防止过度保守
"""

import sys, time, os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic_v2 import ActorCritic, TransformerConfig
from agent.channel_encoder import ChannelEncoder
from agent.ppo import PPOTrainer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class FinetuneConfig:
    def __init__(self, args):
        self.pretrained_dir = args.pretrained_dir
        self.num_frames = args.num_frames
        self.seed = args.seed
        self.device = args.device
        self.snr_db = args.snr
        self.log_interval = 10
        self.frame_len = 512
        self.train_len = 128
        self.pilot_len = 64
        self.num_pilots = 2
        self.num_taps = 16
        self.delay_spread = 10
        self.window_K = 10
        self.state_dim = 45
        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 3
        self.dim_feedforward = 256

        # PPO 微调超参
        self.gamma = 0.95
        self.lam = 0.95
        self.clip_eps = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.actor_lr = 1e-4
        self.k_epochs = 5

        # 冻结策略
        self.freeze_steps = 0  # 不冻结


def load_pretrained(pretrained_dir, device):
    cfg = TransformerConfig(
        state_dim=45, d_model=128, n_heads=4, n_layers=3, dim_feedforward=256,
    )
    ac = ActorCritic(cfg).to(device)
    chan_enc = ChannelEncoder(train_len=128, cond_dim=cfg.cond_dim).to(device)
    ckpt = torch.load(os.path.join(pretrained_dir, "model_best.pt"),
                      map_location=device, weights_only=True)
    ac.load_state_dict(ckpt["ac"])
    chan_enc.load_state_dict(ckpt["chan_enc"])
    print(f"  预训练加载: {pretrained_dir}/model_best.pt")
    return ac, chan_enc


@torch.no_grad()
def eval_ber(ac, chan_enc, env, device, causal=True):
    """评估 BER."""
    states = env.get_all_states().unsqueeze(0).to(device)
    true_bits = env.get_true_bits().to(device)
    rx_all = env.get_rx_symbols().to(device)
    rx_train = rx_all[:128].unsqueeze(0)
    known_bits = env._bits[:128].to(device).unsqueeze(0)
    known_bits_sym = (1 - 2 * known_bits)
    cond = chan_enc(rx_train, known_bits_sym)

    L = states.shape[1]
    mask = None
    if causal:
        mask = torch.triu(torch.ones(L, L, device=device) * float("-inf"), diagonal=1)
    probs, _, _ = ac.forward(states, mask=mask, cond=cond)
    preds = (probs.squeeze(-1) > 0.5).float()
    ber = (preds != true_bits).float().mean().item()
    return ber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_dir", type=str, default="pretrained_nc")
    parser.add_argument("--num_frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = FinetuneConfig(args)

    ac, chan_enc = load_pretrained(cfg.pretrained_dir, device)

    # 独立优化器: 不同部分不同学习率
    optimizer = torch.optim.AdamW([
        {"params": ac.parameters(), "lr": cfg.actor_lr},
        {"params": chan_enc.parameters(), "lr": cfg.actor_lr * 0.5},
    ], weight_decay=1e-5)

    # 环境
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=16, delay_spread=10, snr_db=cfg.snr_db, seed=cfg.seed),
        window_K=10, seed=cfg.seed,
    ))
    env.reset()
    fixed_taps = env.channel._taps.clone()

    # 评估预训练模型
    ac.eval(); chan_enc.eval()
    nc_ber = eval_ber(ac, chan_enc, env, device, causal=False)
    c_ber = eval_ber(ac, chan_enc, env, device, causal=True)
    ac.train(); chan_enc.train()
    print(f"  预训练: 非因果 BER={nc_ber:.5f} | 因果 BER={c_ber:.5f}")

    print(f"\n{'=' * 65}")
    print("  在线因果微调 (信道编码器 + PPO)")
    print(f"  SNR={cfg.snr_db}dB | {cfg.num_frames} frames")
    print(f"{'=' * 65}\n")

    all_bers = []
    all_causal_bers = []
    t_start = time.time()
    best_ber = 1.0

    for fi in range(1, cfg.num_frames + 1):
        # 重置一帧
        env.channel.set_taps(fixed_taps)
        env._bits = env.frame_gen.generate(env._rng)
        env._tx_symbols = env.frame_gen.modulate(env._bits)
        env._rx_symbols = env.channel.convolve(env._tx_symbols)
        env._rx_symbols = env.channel.add_awgn(env._rx_symbols)

        # 信道编码器提取 cond
        rx_all = env.get_rx_symbols().to(device)
        rx_train = rx_all[:128].unsqueeze(0)
        known_bits = env._bits[:128].to(device).unsqueeze(0)
        known_bits_sym = (1 - 2 * known_bits)

        with torch.no_grad():
            cond = chan_enc(rx_train, known_bits_sym)

        # 整帧状态
        states = env.get_all_states().unsqueeze(0).to(device)
        true_bits = env.get_true_bits().to(device)

        # 因果掩码前向
        L = states.shape[1]
        causal_mask = torch.triu(torch.ones(L, L, device=device) * float("-inf"), diagonal=1)
        probs, values, logits = ac.forward(states, mask=causal_mask, cond=cond)

        # 评估
        preds = (probs.squeeze(-1) > 0.5).float()  # (1, 512)
        ber = (preds[0] != true_bits).float().mean().item()
        all_bers.append(ber)

        # 每 50 帧评估一次纯因果 BER (无 PPO 影响)
        if fi % 50 == 0:
            ac.eval(); chan_enc.eval()
            cb = eval_ber(ac, chan_enc, env, device, causal=True)
            ac.train(); chan_enc.train()
            all_causal_bers.append(cb)

        # 构建 PPO 轨迹
        # 奖励: 导频位置 ±1, 训练位置也给奖励, 数据段 0
        bit_types = [env.frame_cfg.bit_type(t) for t in range(cfg.frame_len)]
        known_mask = torch.tensor(
            [bt in ("train", "pilot") for bt in bit_types],
            device=device, dtype=torch.bool,
        )
        correct = (preds[0] == true_bits).float()
        rewards = torch.where(known_mask, 2.0 * correct - 1.0, torch.zeros_like(correct))

        dones = torch.zeros(cfg.frame_len, device=device)
        dones[-1] = 1.0

        # 采样动作
        p_clamped = probs.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        bd = dist.Bernoulli(probs=p_clamped)
        actions = bd.sample().unsqueeze(-1)  # (512, 1)
        log_probs = bd.log_prob(actions.squeeze(-1))

        # PPO 更新 (detach 所有旧数据)
        ppo = PPOTrainer(
            ac, lr=cfg.actor_lr, gamma=cfg.gamma, lam=cfg.lam,
            clip_eps=cfg.clip_eps, value_coef=cfg.value_coef,
            ent_coef=cfg.entropy_coef, max_grad_norm=cfg.max_grad_norm,
        )
        # 替换优化器为我们的 (包含 chan_enc)
        ppo.optimizer = optimizer

        kl_threshold = max(0.005, 0.02 * (1 - fi / cfg.num_frames))
        ppo.train_on_trajectory(
            states=states.squeeze(0).detach(),
            actions=actions.unsqueeze(-1).detach(),
            log_probs=log_probs.squeeze(0).detach(),
            rewards=rewards.detach(),
            values=values.squeeze(-1).squeeze(0).detach(),
            dones=dones.detach(),
            k_epochs=cfg.k_epochs,
            kl_threshold=kl_threshold,
        )

        if ber < best_ber:
            best_ber = ber

        if fi % cfg.log_interval == 0 or fi == 1:
            elapsed = time.time() - t_start
            avg_ber = np.mean(all_bers[-cfg.log_interval:])
            known_acc = (preds[0][known_mask] == true_bits[known_mask]).float().mean().item()
            print(f"  [{fi:4d}/{cfg.num_frames}] BER={ber:.5f}(avg={avg_ber:.5f}) "
                  f"known_acc={known_acc:.3f} best={best_ber:.5f} ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    final_ber = np.mean(all_bers[-20:])
    print(f"\n{'=' * 65}")
    print(f"  在线微调完成! ({elapsed:.0f}s)")
    print(f"  Best BER: {best_ber:.5f}")
    print(f"  Final avg BER (last 20): {final_ber:.5f}")
    print(f"{'=' * 65}")

    # 保存
    os.makedirs("finetuned", exist_ok=True)
    torch.save({
        "ac": ac.state_dict(),
        "chan_enc": chan_enc.state_dict(),
        "best_ber": best_ber,
        "final_ber": final_ber,
    }, "finetuned/model_best.pt")

    # 绘图
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        xs = range(1, len(all_bers) + 1)
        ax1.plot(xs, all_bers, alpha=0.3, color="#1565C0", lw=0.5)
        if len(all_bers) >= 21:
            s = np.convolve(all_bers, np.ones(21)/21, mode="valid")
            ax1.plot(range(21, len(all_bers) + 1), s, color="#0D47A1", lw=2)
        ax1.axhline(0.01, color="#FF9800", ls="--", alpha=0.5)
        ax1.set_ylabel("BER (online)"); ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        if all_causal_bers:
            ax2.plot(range(50, 50 * len(all_causal_bers) + 1, 50),
                     all_causal_bers, "o-", color="#E53935")
            ax2.set_ylabel("BER (causal eval)"); ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

        ax2.set_xlabel("Frame")
        plt.suptitle(f"Online Finetune (SNR={cfg.snr_db}dB)")
        plt.tight_layout()
        plt.savefig("finetuned/finetune_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  图保存到 finetuned/finetune_curve.png")

    return best_ber


if __name__ == "__main__":
    main()

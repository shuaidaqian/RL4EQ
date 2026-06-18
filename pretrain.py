# -*- coding: utf-8 -*-
"""
离线预训练 — 固定信道监督学习均衡器初始化。

核心思路:
  在固定信道上对 Actor-Critic 网络的 Actor 部分做监督预训练，
  让网络先学会该信道的 ISI 均衡模式。
  然后用 --finetune 在 online_train.py 中做 PPO 微调适应新信道。

  注意: 块衰落模式下 45 维状态信息不足无法收敛，
        所以预训练使用固定信道，在线微调负责信道适应。

用法:
  python pretrain.py                                    # 默认参数预训练
  python pretrain.py --snr 10 --num_steps 1000          # 自定义训练步数
  python pretrain.py --snr 5  --num_steps 2000          # 低 SNR 更大训练量
  python pretrain.py --save_dir pretrained              # 输出目录

在线微调:
  python online_train.py --finetune pretrained/actor_critic_best.pt --snr 5
"""

import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig, FrameGenerator
from env.channel_models import RayleighMultipathChannel
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic import ActorCritic, TransformerConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def create_env(snr_db, seed):
    """创建一次性的通信环境。"""
    return CommunicationEnv(EnvConfig(
        frame=FrameConfig(frame_len=512, train_len=128, pilot_len=64, num_pilots=2),
        channel=dict(num_taps=16, delay_spread=10, snr_db=snr_db, seed=seed),
        window_K=10, seed=seed,
    ))


@torch.no_grad()
def generate_batch(envs, frames_per_env=3):
    """
    从多个环境中各生成多帧数据，组成混合批。
    每个环境（信道）生成多帧，让网络看到同一 ISI 模式重复出现。

    返回:
        states:    (B, L, D)  所有帧状态
        known_mask: (B, L)    已知位 mask
        true_bits:  (B, L)    真实比特
    """
    batch_states, batch_known, batch_bits = [], [], []
    for env in envs:
        for _ in range(frames_per_env):
            env.channel.reset_taps()
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)
            env._t = 0
            env._done = False

            states = env.get_all_states()
            true_bits = env.get_true_bits()
            mask = torch.tensor(
                [env.frame_cfg.bit_type(t) in ("train", "pilot")
                 for t in range(env.frame_cfg.frame_len)],
                dtype=torch.bool,
            )
            batch_states.append(states)
            batch_known.append(mask)
            batch_bits.append(true_bits)

    return (
        torch.stack(batch_states),     # (B, L, D)
        torch.stack(batch_known),      # (B, L)
        torch.stack(batch_bits),       # (B, L)
    )


def compute_supervised_loss(ac, states, known_mask, true_bits):
    """
    计算监督损失：只在已知位算 BCE。

    参数:
        ac: ActorCritic 网络
        states:     (B, L, D)
        known_mask: (B, L) bool
        true_bits:  (B, L) float
    """
    B, L, _ = states.shape
    mask = ac._get_causal_mask(L, states.device)
    _, _, logits = ac.forward(states, mask=mask)  # (B, L, 1)
    logits = logits.squeeze(-1)  # (B, L)

    if known_mask.any():
        loss = F.binary_cross_entropy_with_logits(
            logits[known_mask], true_bits[known_mask],
        )
    else:
        loss = torch.zeros(1, device=states.device)

    # 准确率（监控）
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds[known_mask] == true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0
        ber = (preds != true_bits).float().mean().item()

    return loss, acc, ber


def compute_pretrain_loss(ac, states, known_mask, true_bits):
    """
    预训练损失：先最小化已知位 BCE，再传播到数据位。

    - 使用全帧双向注意力（预训练阶段不需要因果）
    - 仅对已知位计算 BCE 损失
    - 数据位通过共享 Transformer 参数间接学习
    """
    B, L, D = states.shape
    device = states.device
    x = ac.input_proj(states)
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    x = x + ac.pos_encoding(pos)
    # 无 mask = 全帧双向注意力
    x = ac.transformer(x)
    # 只训练 Actor 部分
    logits = ac.actor_fc(x).squeeze(-1)  # (B, L)

    if known_mask.any():
        loss = F.binary_cross_entropy_with_logits(
            logits[known_mask], true_bits[known_mask],
        )
    else:
        loss = torch.zeros(1, device=device)

    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds[known_mask] == true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0
        ber = (preds != true_bits).float().mean().item()

    return loss, acc, ber


def smooth_curve(data, window=21):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 网络
    tf_cfg = TransformerConfig(state_dim=45, d_model=64, n_heads=4,
                               n_layers=2, dim_feedforward=128)
    ac = ActorCritic(tf_cfg).to(device)
    optimizer = torch.optim.AdamW(ac.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    params = sum(p.numel() for p in ac.parameters())
    print("=" * 65)
    print("  RL4EQ — 离线预训练 (固定信道监督学习)")
    print("=" * 65)
    print(f"  网络: Transformer {tf_cfg.n_layers}L {tf_cfg.n_heads}H")
    print(f"  参数量: {params:,}")
    print(f"  训练步数: {args.num_steps}")
    print(f"  SNR: {args.snr}dB")
    print(f"  学习率: {args.lr}")
    print()

    # 使用同一个固定信道训练（在线微调负责信道适应）
    base_env = create_env(args.snr, seed=42)
    base_env.reset()
    fixed_taps = base_env.channel._taps.clone()

    all_losses = []
    all_accs = []
    all_bers = []
    t_start = time.time()
    best_val_ber = 1.0

    for step in range(1, args.num_steps + 1):
        # 使用固定信道，每步生成新训练数据
        base_env.channel.set_taps(fixed_taps)
        base_env._bits = base_env.frame_gen.generate(base_env._rng)
        base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)

        states = base_env.get_all_states().unsqueeze(0).to(device)  # (1, L, D)
        true_bits = base_env.get_true_bits().unsqueeze(0).to(device)
        mask_t = torch.tensor(
            [base_env.frame_cfg.bit_type(t) in ("train", "pilot")
             for t in range(base_env.frame_cfg.frame_len)],
            device=device, dtype=torch.bool,
        ).unsqueeze(0)  # (1, L)

        # 前向
        ac.train()
        loss, acc, ber = compute_pretrain_loss(ac, states, mask_t, true_bits)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        all_losses.append(loss.item())
        all_accs.append(acc)
        all_bers.append(ber)

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - t_start
            avg_loss = np.mean(all_losses[-100:])
            avg_acc = np.mean(all_accs[-100:])
            print(f"  Step {step:5d}/{args.num_steps}"
                  f" | loss={avg_loss:.4f}"
                  f" | known_acc={avg_acc:.3f}"
                  f" | BER={np.mean(all_bers[-100:]):.4f}"
                  f" | lr={scheduler.get_last_lr()[0]:.2e}"
                  f" | {elapsed:.1f}s")

        # 每 500 步验证（同一信道新数据）
        if step % 500 == 0:
            ac.eval()
            with torch.no_grad():
                base_env.channel.set_taps(fixed_taps)
                base_env._bits = base_env.frame_gen.generate(base_env._rng)
                base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
                base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
                base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
                vs = base_env.get_all_states().unsqueeze(0).to(device)
                vt = base_env.get_true_bits().unsqueeze(0).to(device)
                vm = torch.tensor(
                    [base_env.frame_cfg.bit_type(t) in ("train", "pilot")
                     for t in range(base_env.frame_cfg.frame_len)],
                    device=device, dtype=torch.bool,
                ).unsqueeze(0)
                _, _, v_logits = ac.forward(vs)
                v_preds = (torch.sigmoid(v_logits.squeeze(-1)) > 0.5).float()
                v_acc = (v_preds[vm] == vt[vm]).float().mean().item() if vm.any() else 0.0
                v_ber = (v_preds != vt).float().mean().item()
            print(f"  Val[{step:5d}] known_acc={v_acc:.3f} BER={v_ber:.5f}")
            if v_ber < best_val_ber:
                best_val_ber = v_ber
                torch.save(ac.state_dict(), os.path.join(save_dir, "actor_critic_best.pt"))
                print(f"  >>> 保存最佳模型 BER={v_ber:.5f}")

    # 最终保存
    torch.save(ac.state_dict(), os.path.join(save_dir, "actor_critic_final.pt"))
    elapsed = time.time() - t_start

    print(f"\n{'=' * 65}")
    print(f"  预训练完成! ({elapsed:.1f}s)")
    print(f"  Best val BER: {best_val_ber:.5f}")
    print(f"  模型保存至: {save_dir}/")
    print(f"{'=' * 65}")

    # 绘图
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        steps = np.arange(1, len(all_losses) + 1)
        ax1.plot(steps, all_losses, alpha=0.3, color="#2196F3", lw=0.5)
        if len(all_losses) >= 21:
            sl = smooth_curve(all_losses, 21)
            ax1.plot(range(21, len(all_losses) + 1), sl, color="#1565C0", lw=2)
        ax1.set_ylabel("BCE Loss")
        ax1.set_title("Pretraining Loss Curve")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, all_bers, alpha=0.3, color="#F44336", lw=0.5)
        if len(all_bers) >= 21:
            sb = smooth_curve(all_bers, 21)
            ax2.plot(range(21, len(all_bers) + 1), sb, color="#C62828", lw=2)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("BER")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pretrain_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    return best_val_ber


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
因果预训练 — 使用因果掩码 + 信道编码器 cond

核心思路:
  - 使用与在线推理完全一致的因果掩码
  - 信道编码器从训练序列提取 cond (信道特征), 辅助预测
  - 每个时间步只能看到过去的信息 + 全局信道特征
  - 这样预训练和在线微调之间没有 gap

相比非因果方案:
  ✓ 预训练和在线推理完全一致
  ✓ 不用学"未来信息", 模型更专注 ISI 模式
  ✓ 需要更多步数收敛, 但不依赖双向注意力
"""

import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic_v2 import ActorCritic, TransformerConfig
from agent.channel_encoder import ChannelEncoder

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def smooth_curve(data, w=21):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default="pretrained_causal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--snr", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 网络
    tf_cfg = TransformerConfig(state_dim=45, d_model=args.d_model, n_heads=4,
                               n_layers=args.n_layers, dim_feedforward=args.ffn)
    ac = ActorCritic(tf_cfg).to(device)
    chan_enc = ChannelEncoder(train_len=128, cond_dim=tf_cfg.cond_dim).to(device)
    params = sum(p.numel() for p in ac.parameters()) + sum(p.numel() for p in chan_enc.parameters())

    print("=" * 65)
    print("  因果预训练 (因果掩码 + 信道编码器 cond)")
    print(f"  参数量: {params} | d_model={args.d_model} | layers={args.n_layers}")
    print()

    optimizer = torch.optim.AdamW(
        list(ac.parameters()) + list(chan_enc.parameters()),
        lr=5e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=5e-5
    )

    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=16, delay_spread=10, snr_db=args.snr, seed=args.seed),
        window_K=10, seed=args.seed,
    ))
    env.reset()
    fixed_taps = env.channel._taps.clone()

    all_bers = []
    t_start = time.time()
    best_ber = 1.0
    train_len = 128
    L = 512

    for step in range(1, args.num_steps + 1):
        # 每步跑 2 帧累计梯度
        for _ in range(2):
            env.channel.set_taps(fixed_taps)
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)

            # 信道编码器
            rx_all = env.get_rx_symbols().to(device)
            rx_train = rx_all[:train_len].unsqueeze(0)
            known_bits = env._bits[:train_len].to(device).unsqueeze(0)
            known_bits_sym = (1 - 2 * known_bits)
            cond = chan_enc(rx_train, known_bits_sym)

            # 状态 + 因果掩码
            states = env.get_all_states().unsqueeze(0).to(device)
            true_bits = env.get_true_bits().to(device)

            causal_mask = ac._get_causal_mask(L, device)
            probs, values, logits = ac.forward(states, mask=causal_mask, cond=cond)

            # 只在已知位置 (训练+导频) 算 loss
            known_mask = torch.tensor(
                [env.frame_cfg.bit_type(t) in ("train", "pilot") for t in range(L)],
                device=device, dtype=torch.bool,
            )
            logits_flat = logits.squeeze(-1)  # (1, 512)
            loss = F.binary_cross_entropy_with_logits(
                logits_flat[0, known_mask], true_bits[known_mask]
            )
            # 额外监督 critic
            with torch.no_grad():
                correct = ((torch.sigmoid(logits_flat) > 0.5).float() == true_bits.unsqueeze(0)).float()
                target_v = torch.where(known_mask.unsqueeze(0), correct * 2 - 1, torch.zeros_like(correct))
            loss += 0.2 * F.mse_loss(values.squeeze(-1), target_v)

            (loss / 2.0).backward()

        nn.utils.clip_grad_norm_(
            list(ac.parameters()) + list(chan_enc.parameters()), 1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits_flat) > 0.5).float()
            ber = (preds[0] != true_bits).float().mean().item()
        all_bers.append(ber)

        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t_start
            avg_ber = np.mean(all_bers[-200:])
            print(f"  Step {step:5d}/{args.num_steps} | BER={avg_ber:.4f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # 验证
        if step % 500 == 0 or step == args.num_steps:
            env.channel.set_taps(fixed_taps)
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)

            ac.eval(); chan_enc.eval()
            with torch.no_grad():
                rx_all_v = env.get_rx_symbols().to(device)
                rx_train_v = rx_all_v[:train_len].unsqueeze(0)
                kb_v = env._bits[:train_len].to(device).unsqueeze(0)
                kb_v_sym = (1 - 2 * kb_v)
                cond_v = chan_enc(rx_train_v, kb_v_sym)
                vs = env.get_all_states().unsqueeze(0).to(device)
                vt = env.get_true_bits().to(device)
                cm = ac._get_causal_mask(L, device)
                p_v, _, _ = ac.forward(vs, mask=cm, cond=cond_v)
                pred_v = (p_v.squeeze(-1) > 0.5).float()
                val_ber = (pred_v[0] != vt).float().mean().item()
            ac.train(); chan_enc.train()
            print(f"  Val[{step:5d}] BER={val_ber:.5f}")
            if val_ber < best_ber:
                best_ber = val_ber
                torch.save({
                    "ac": ac.state_dict(),
                    "chan_enc": chan_enc.state_dict(),
                }, os.path.join(save_dir, "model_best.pt"))
                print(f"  >>> 保存最佳模型 BER={val_ber:.5f}")

    torch.save({"ac": ac.state_dict(), "chan_enc": chan_enc.state_dict()},
               os.path.join(save_dir, "model_final.pt"))
    elapsed = time.time() - t_start
    print()
    print("=" * 65)
    print(f"  因果预训练完成! ({elapsed:.0f}s)")
    print(f"  Best val BER: {best_ber:.5f}")

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(range(1, len(all_bers) + 1), all_bers, alpha=0.3, color="#F44336", lw=0.5)
        if len(all_bers) >= 21:
            ax.plot(range(21, len(all_bers) + 1), smooth_curve(all_bers, 21), color="#C62828", lw=2)
        ax.axhline(0.05, color="#FF9800", ls="--", alpha=0.5, label="5%")
        ax.set_ylabel("BER")
        ax.set_xlabel("Step")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pretrain_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print("=" * 65)
    return best_ber


if __name__ == "__main__":
    main()

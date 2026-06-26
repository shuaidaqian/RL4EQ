# -*- coding: utf-8 -*-
"""
离线预训练 V6 — 固定信道因果预训练 (最优初始化)

核心策略:
  - 固定信道 + SNR=5
  - 因果掩码 (与微调一致)
  - Actor BCE + Critic 预训练
  - 1000 步, 余弦退火, 多帧累积

用法:
  python pretrain.py
  python online_train.py --finetune pretrained/actor_critic_best.pt --snr 5 --num_frames 300
"""

import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic import ActorCritic, TransformerConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def compute_loss(ac, states, known_mask, true_bits, cw=0.0):
    B, L = states.shape[0], states.shape[1]
    device = states.device
    mask = ac._get_causal_mask(L, device)
    x = ac.input_proj(states) + ac.pos_encoding(torch.arange(L, device=device).unsqueeze(0).expand(B, -1))
    x = ac.transformer(x, mask=mask)
    logits = ac.actor_fc(x).squeeze(-1)
    actor_loss = F.binary_cross_entropy_with_logits(logits[known_mask], true_bits[known_mask]) if known_mask.any() else torch.zeros(1, device=device)
    if cw > 0:
        values = ac.critic(x).squeeze(-1)
        with torch.no_grad():
            correct = (torch.sigmoid(logits) > 0.5).float() == true_bits
            clabel = torch.where(known_mask, 2.0*correct.float()-1.0, torch.zeros_like(correct.float())) * 0.5
        total_loss = actor_loss + cw * F.mse_loss(values, clabel)
    else:
        total_loss = actor_loss
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds[known_mask]==true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0
        ber = (preds!=true_bits).float().mean().item()
    return total_loss, actor_loss.item(), acc, ber


def smooth_curve(data, w=21):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    tf_cfg = TransformerConfig(state_dim=45, d_model=args.d_model, n_heads=4,
                               n_layers=args.n_layers, dim_feedforward=args.ffn)
    ac = ActorCritic(tf_cfg).to(device)
    params = sum(p.numel() for p in ac.parameters())

    print("="*65)
    print("  RL4EQ — 离线预训练 V6 (固定信道)")
    print("="*65)
    print(f"  参数量: {params}  步数: {args.num_steps}")
    print(f"  固定信道 SNR=5  因果掩码  Actor+Critic")
    print()

    optimizer = torch.optim.AdamW(ac.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=1e-5)

    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(frame_len=512, train_len=128, pilot_len=64, num_pilots=2),
        channel=dict(num_taps=16, delay_spread=10, snr_db=5.0, seed=args.seed),
        window_K=10, seed=args.seed,
    ))
    env.reset()
    fixed_taps = env.channel._taps.clone()

    all_bers = []
    t_start = time.time()
    best_ber = 1.0

    for step in range(1, args.num_steps + 1):
        cw = 0.1 if step > 300 else 0.0
        for _ in range(3):
            env.channel.set_taps(fixed_taps)
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)
            states = env.get_all_states().unsqueeze(0).to(device)
            true_bits = env.get_true_bits().unsqueeze(0).to(device)
            mask_t = torch.tensor(
                [env.frame_cfg.bit_type(t) in ("train","pilot") for t in range(512)],
                device=device, dtype=torch.bool,
            ).unsqueeze(0)
            loss, al, acc, ber = compute_loss(ac, states, mask_t, true_bits, cw)
            (loss / 3.0).backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        all_bers.append(ber)

        if step % 100 == 0 or step == 1:
            el = time.time() - t_start
            print(f"  Step {step:5d}/{args.num_steps} | BER={np.mean(all_bers[-100:]):.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {el:.0f}s")

        if step % 250 == 0:
            env.channel.set_taps(fixed_taps)
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)
            ac.eval()
            with torch.no_grad():
                vs = env.get_all_states().unsqueeze(0).to(device)
                vt = env.get_true_bits().unsqueeze(0).to(device)
                cm = ac._get_causal_mask(512, device)
                x = ac.input_proj(vs) + ac.pos_encoding(torch.arange(512, device=device).unsqueeze(0))
                x = ac.transformer(x, mask=cm)
                v_l = ac.actor_fc(x).squeeze(-1)
                v_p = (torch.sigmoid(v_l) > 0.5).float()
                al_ber = (v_p != vt).float().mean().item()
            ac.train()
            print(f"  Val[{step:5d}] BER={al_ber:.5f}")
            if al_ber < best_ber:
                best_ber = al_ber
                torch.save(ac.state_dict(), os.path.join(save_dir, "actor_critic_best.pt"))
                print(f"  >>> 保存最佳模型 BER={al_ber:.5f}")

    torch.save(ac.state_dict(), os.path.join(save_dir, "actor_critic_final.pt"))
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  预训练 V6 完成! ({elapsed:.0f}s)")
    print(f"  Best val BER: {best_ber:.5f} (固定信道)")

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(range(1, len(all_bers)+1), all_bers, alpha=0.3, color="#F44336", lw=0.5)
        if len(all_bers) >= 21:
            ax.plot(range(21, len(all_bers)+1), smooth_curve(all_bers, 21), color="#C62828", lw=2)
        ax.axhline(0.05, color="#FF9800", ls="--", alpha=0.5, label="5%")
        ax.set_ylabel("BER")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pretrain_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    return best_ber


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
非因果预训练 — 使用双向注意力 (看到整帧)

策略:
  - 双向 Transformer 编码器 (无因果掩码)
  - 每帧预测所有比特 (训练+导频+数据)
  - 固定信道 + SNR=5 预训练, 让网络学习 ISI 模式
  - 保存模型用于后续在线因果微调

用法:
  python pretrain_noncausal.py
  python online_train.py --finetune pretrained_nc/actor_critic_best.pt
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


def compute_loss(ac, chan_enc, states, rx_train, known_bits, known_mask, true_bits, cw=0.0, use_noncausal=True):
    """非因果/因果训练损失
    use_noncausal=True: 双向注意力, 整帧预测
    use_noncausal=False: 因果掩码 (与在线一致)
    """
    B, L = states.shape[0], states.shape[1]
    device = states.device

    # 信道条件嵌入
    cond = chan_enc(rx_train, known_bits)

    # 前向传播
    x = ac.input_proj(states) + ac.pos_encoding(torch.arange(L, device=device).unsqueeze(0).expand(B, -1))
    if use_noncausal:
        x = ac.transformer(x)  # 无掩码 = 双向注意力
    else:
        mask = ac._get_causal_mask(L, device)
        x = ac.transformer(x, mask=mask)

    # FiLM 条件调制 (用信道编码器的嵌入替换 cond_net 自提取)
    for film in ac.film_layers:
        x = film(x, cond)

    logits = ac.actor_fc(x).squeeze(-1)
    values = ac.critic(x).squeeze(-1)

    # Actor loss (所有位置, 非因果下所有位置都有预测)
    actor_loss = F.binary_cross_entropy_with_logits(
        logits[known_mask], true_bits[known_mask]) if known_mask.any() else torch.zeros(1, device=device)

    # Critic loss
    if cw > 0:
        with torch.no_grad():
            correct = (torch.sigmoid(logits) > 0.5).float() == true_bits
            clabel = torch.where(known_mask, 2.0 * correct.float() - 1.0, torch.zeros_like(correct.float())) * 0.5
        total_loss = actor_loss + cw * F.mse_loss(values, clabel)
    else:
        total_loss = actor_loss

    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds[known_mask] == true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0
        ber = (preds != true_bits).float().mean().item()

    return total_loss, actor_loss.item(), acc, ber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="pretrained_nc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--noncausal", action="store_true", default=True)
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
    print("  非因果预训练 (双向注意力 + 信道编码器)")
    print("=" * 65)
    print(f"  参数量: {params} | 非因果={args.noncausal}")
    print()

    optimizer = torch.optim.AdamW(
        list(ac.parameters()) + list(chan_enc.parameters()),
        lr=3e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=1e-5
    )

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
    train_len = 128

    for step in range(1, args.num_steps + 1):
        cw = 0.1 if step > 300 else 0.0
        for _ in range(3):
            env.channel.set_taps(fixed_taps)
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)

            # 获取训练序列
            rx_all = env.get_rx_symbols().to(device)  # (512, 2)
            rx_train = rx_all[:train_len].unsqueeze(0)  # (1, 128, 2)
            known_bits = env._bits[:train_len].to(device).unsqueeze(0)  # (1, 128)
            known_bits_sym = (1 - 2 * known_bits)  # 0->1, 1->-1

            states = env.get_all_states().unsqueeze(0).to(device)
            true_bits = env.get_true_bits().unsqueeze(0).to(device)
            mask_t = torch.tensor(
                [env.frame_cfg.bit_type(t) in ("train", "pilot") for t in range(512)],
                device=device, dtype=torch.bool,
            ).unsqueeze(0)

            loss, al, acc, ber = compute_loss(
                ac, chan_enc, states, rx_train, known_bits_sym, mask_t, true_bits, cw,
                use_noncausal=args.noncausal
            )
            (loss / 3.0).backward()

        nn.utils.clip_grad_norm_(
            list(ac.parameters()) + list(chan_enc.parameters()), 1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        all_bers.append(ber)

        if step % 100 == 0 or step == 1:
            el = time.time() - t_start
            print(f"  Step {step:5d}/{args.num_steps} | BER={np.mean(all_bers[-100:]):.4f}",
                  f"| lr={scheduler.get_last_lr()[0]:.2e} | {el:.0f}s")

        if step % 250 == 0:
            env.channel.set_taps(fixed_taps)
            env._bits = env.frame_gen.generate(env._rng)
            env._tx_symbols = env.frame_gen.modulate(env._bits)
            env._rx_symbols = env.channel.convolve(env._tx_symbols)
            env._rx_symbols = env.channel.add_awgn(env._rx_symbols)
            ac.eval(); chan_enc.eval()
            with torch.no_grad():
                rx_all_val = env.get_rx_symbols().to(device)
                rx_train_val = rx_all_val[:train_len].unsqueeze(0)
                kb_val = env._bits[:train_len].to(device).unsqueeze(0)
                kb_val_sym = (1 - 2 * kb_val)
                vs = env.get_all_states().unsqueeze(0).to(device)
                vt = env.get_true_bits().unsqueeze(0).to(device)
                cond_v = chan_enc(rx_train_val, kb_val_sym)
                x = ac.input_proj(vs) + ac.pos_encoding(torch.arange(512, device=device).unsqueeze(0))
                x = ac.transformer(x)  # 非因果验证
                for film in ac.film_layers:
                    x = film(x, cond_v)
                v_l = ac.actor_fc(x).squeeze(-1)
                v_p = (torch.sigmoid(v_l) > 0.5).float()
                al_ber = (v_p != vt).float().mean().item()
            ac.train(); chan_enc.train()
            print(f"  Val[{step:5d}] BER={al_ber:.5f}")
            if al_ber < best_ber:
                best_ber = al_ber
                torch.save({
                    "ac": ac.state_dict(),
                    "chan_enc": chan_enc.state_dict(),
                }, os.path.join(save_dir, "model_best.pt"))
                print(f"  >>> 保存最佳模型 BER={al_ber:.5f}")

    torch.save({"ac": ac.state_dict(), "chan_enc": chan_enc.state_dict()},
               os.path.join(save_dir, "model_final.pt"))
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  非因果预训练完成! ({elapsed:.0f}s)")
    print(f"  Best val BER: {best_ber:.5f}")

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(range(1, len(all_bers) + 1), all_bers, alpha=0.3, color="#F44336", lw=0.5)
        if len(all_bers) >= 21:
            ax.plot(range(21, len(all_bers) + 1), smooth_curve(all_bers, 21), color="#C62828", lw=2)
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

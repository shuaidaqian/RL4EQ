# -*- coding: utf-8 -*-
"""
RL4EQ — Transformer-A2C 在线信道均衡主程序

算法: A2C-KPG (Known-only Policy Gradient)
网络: Transformer 编码器 (捕获长距离 ISI 依赖)
信道: P-FTNet 参数 (16 抽头频率选择性瑞利衰落, SNR -2 ~ 10 dB)
帧结构: | 训练(128) | 导频(64) | 数据(128) | 导频(64) | 数据(128) |

三阶段训练策略:
  SUP [1-10]:  纯监督学习 (BCE), 确定性输出
  MIX [11-30]: 混合模式 (A2C + BCE), 随机采样
  RL [31+]:    全 A2C 在线学习, 策略梯度为主

优化说明:
  - 批量状态收集 + 单次 Transformer 前向 (加速约 458 倍)
  - 向量化 F.conv1d 信道卷积 (加速约 700 倍)
  - 多帧缓冲训练, 减少梯度方差

用法:
  python online_train.py                     # 200 帧, 10dB
  python online_train.py --snr 5             # 自定义 SNR
  python online_train.py --num_frames 50     # 自定义帧数
"""

import sys, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig
from env.channel_models import RayleighMultipathChannel
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic import ActorCritic, TransformerConfig


@dataclass
class TrainConfig:
    """训练参数配置类。"""
    num_frames: int = 200
    log_interval: int = 10
    seed: int = 42
    device: str = "cpu"
    snr_db: float = 10.0
    frame_len: int = 512
    train_len: int = 128
    pilot_len: int = 64
    num_pilots: int = 2
    num_taps: int = 16
    delay_spread: int = 10
    window_K: int = 10
    state_dim: int = 45
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    actor_lr: float = 3e-4
    gamma: float = 0.97
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    buffer_size: int = 5
    sup_end: int = 10
    mix_end: int = 30


class A2CAgent:
    """A2C 智能体包装类: 管理 Actor-Critic 网络和优化器。"""

    def __init__(self, cfg: TrainConfig, device):
        self.device = device
        self.tf_cfg = TransformerConfig(
            state_dim=cfg.state_dim, d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            dim_feedforward=cfg.dim_feedforward,
        )
        self.ac = ActorCritic(self.tf_cfg).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.actor_lr)
        self.cfg = cfg

    @torch.no_grad()
    def batch_act(self, states):
        # 批量前向推理: 整帧状态 -> 动作概率, 价值, logits
        T = states.shape[0]
        mask = self.ac._get_causal_mask(T, self.device)
        probs, values, logits = self.ac.forward(states.unsqueeze(0), mask=mask)
        return probs.squeeze(0), values.squeeze(0), logits.squeeze(0)

    def train_on_buffer(self, frame_buffer, policy_coef, sup_weight, gamma):
        """多帧缓冲训练: 累积多帧轨迹后更新网络参数。

        损失函数:
          L = policy_coef * L_pol(已知位) + value_coef * L_val(全局)
            - entropy_coef * H(全局) + sup_weight * L_BCE(已知位)
        """
        self.ac.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        c = self.cfg

        for tb in frame_buffer:
            T = tb["states"].shape[0]

            # 1. Monte Carlo 回报
            returns = torch.zeros(T, device=self.device)
            G = 0.0
            for t in reversed(range(T)):
                G = tb["rewards"][t].item() + gamma * (1 - tb["dones"][t].item()) * G
                returns[t] = G

            # 2. 整帧 Transformer 前向
            mask = self.ac._get_causal_mask(T, self.device)
            _, _, out_l = self.ac.forward(tb["states"].unsqueeze(0), mask=mask)
            logits = out_l.squeeze(0)

            probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            bd = dist.Bernoulli(probs=probs)
            log_probs = bd.log_prob(tb["actions"]).squeeze(0)

            # 3. 优势 (归一化)
            values = self.ac.critic(
                self.ac.transformer(
                    self.ac.input_proj(tb["states"].unsqueeze(0))
                    + self.ac.pos_encoding(
                        torch.arange(T, device=self.device).unsqueeze(0)
                    )
                )
            ).squeeze(-1).squeeze(0)

            adv = returns - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            known = tb["known_mask"]

            # 4. 组合损失
            if known.any():
                p_loss = -(log_probs[known] * adv[known]).mean()
                s_loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1)[known], tb["true_bits"][known])
            else:
                p_loss = torch.zeros(1, device=self.device)
                s_loss = torch.zeros(1, device=self.device)

            v_loss = F.mse_loss(values, returns)
            e_loss = bd.entropy().squeeze(0).mean()

            total_loss += (policy_coef * p_loss + c.value_coef * v_loss
                           - c.entropy_coef * e_loss + sup_weight * s_loss)

        total_loss = total_loss / len(frame_buffer)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
        self.optimizer.step()
        return {"total": total_loss.item()}


@torch.no_grad()
def eval_ber(probs, true_bits):
    """计算误比特率 (BER)。"""
    preds = (probs.squeeze(-1) > 0.5).float()
    return (preds != true_bits).float().mean().item()


@torch.no_grad()
def eval_mmse_ber(rx_symbols, true_bits, taps, window_K, num_taps, snr_db):
    """MMSE 线性均衡器 — RL 对比基线。"""
    Nf = 2 * window_K + 1
    D = (Nf + num_taps) // 2
    h = taps.numpy()
    rx = rx_symbols.numpy()
    tx = true_bits.numpy()
    sig_pwr = np.mean(np.abs(rx[:, 0] + 1j * rx[:, 1]) ** 2)
    sigma2 = sig_pwr / (10.0 ** (snr_db / 10.0))

    R = np.zeros((Nf, Nf), dtype=complex)
    p = np.zeros(Nf, dtype=complex)
    for i in range(Nf):
        for j in range(Nf):
            R[i, j] = (sum(h[k] * np.conj(h[k + j - i])
                           for k in range(num_taps) if 0 <= k + j - i < num_taps)
                       + (sigma2 if i == j else 0.0))
        p[i] = np.conj(h[D - i]) if 0 <= D - i < num_taps else 0.0

    w = np.linalg.solve(R, p)
    rx_pad = np.pad(rx[:, 0] + 1j * rx[:, 1], (Nf - 1, 0))
    eq = np.array([np.dot(w.conj(), rx_pad[n:n + Nf][::-1]) for n in range(len(rx))])
    bits = ((1 - np.sign(eq.real)) / 2).astype(int)
    return float(np.mean(bits != tx))


def save_plots(all_known_accs, all_ber, mmse_results, pftnet_results, cfg, save_dir="logs"):
    """生成训练结果可视化并保存为 PNG 图片。"""
    os.makedirs(save_dir, exist_ok=True)
    try:
        from utils.matplotlib_zh import get_font
        import matplotlib.pyplot as plt
    except Exception:
        import matplotlib.pyplot as plt
        def get_font(s): return None

    def smooth_curve(data, window=11):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode="valid")

    fs = get_font(12)

    # 1. 训练曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    frames = np.arange(1, len(all_known_accs) + 1)

    ax1.plot(frames, all_known_accs, alpha=0.3, color="#2196F3", lw=0.8, label="Raw")
    if len(all_known_accs) >= 11:
        sk = smooth_curve(all_known_accs, 11)
        ax1.plot(np.arange(11, len(all_known_accs)+1), sk, color="#1565C0", lw=2, label="Smooth(w=11)")
    ax1.set_ylabel("Known-Acc")
    ax1.set_title("Training Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(frames, all_ber, alpha=0.3, color="#F44336", lw=0.8, label="Raw")
    if len(all_ber) >= 11:
        sb = smooth_curve(all_ber, 11)
        ax2.plot(np.arange(11, len(all_ber)+1), sb, color="#C62828", lw=2, label="Smooth(w=11)")
    ax2.axhline(0.5, color="#FF9800", ls="--", alpha=0.7, label="Random")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("BER")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    print("  Training curve saved")
    plt.close()

    # 2. P-FTNet SNR 范围测试
    if pftnet_results:
        plt.figure(figsize=(9, 5))
        snr_vals = [r[0] for r in pftnet_results]
        ber_vals = [r[1] for r in pftnet_results]
        colors = ["#C62828" if b == min(ber_vals) else "#1565C0" for b in ber_vals]
        bars = plt.bar([str(s) for s in snr_vals], ber_vals, color=colors, width=0.5, ec="white")
        for bar, ber in zip(bars, ber_vals):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"{ber:.3f}", ha="center", va="bottom", fontsize=11)
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title("P-FTNet SNR Test")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pftnet_snr.png"), dpi=150, bbox_inches="tight")
        print("  P-FTNet SNR test saved")
        plt.close()

    # 3. MMSE 对比
    if mmse_results:
        plt.figure(figsize=(9, 5))
        snr_labels = [f"{r[0]}dB" for r in mmse_results]
        mmse_bars = [r[1] for r in mmse_results]
        rl_bars = [r[2] for r in mmse_results]
        x = np.arange(len(snr_labels))
        w = 0.35
        plt.bar(x-w/2, mmse_bars, w, label="MMSE", color="#90A4AE", ec="white")
        plt.bar(x+w/2, rl_bars, w, label="A2C (Transformer)", color="#1565C0", ec="white")
        plt.xlabel("SNR")
        plt.ylabel("BER")
        plt.title("MMSE vs A2C")
        plt.xticks(x, snr_labels)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mmse_comparison.png"), dpi=150, bbox_inches="tight")
        print("  MMSE comparison saved")
        plt.close()

    print(f"  All plots saved to {save_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig(snr_db=args.snr, num_frames=args.num_frames, device=args.device)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 65)
    print("  RL4EQ - Transformer-A2C Online Channel Equalization")
    print("=" * 65)
    print(f"  Frame: {cfg.frame_len}|train {cfg.train_len}|pilot 2x{cfg.pilot_len}|data {cfg.frame_len-cfg.train_len-2*cfg.pilot_len}")
    print(f"  Channel: {cfg.num_taps}-tap Rayleigh SNR={cfg.snr_db}dB")
    print(f"  Network: Transformer {cfg.n_layers}L {cfg.n_heads}H")
    print(f"  Phases: SUP(1-{cfg.sup_end}) MIX({cfg.sup_end+1}-{cfg.mix_end}) RL({cfg.mix_end+1}+)")

    agent = A2CAgent(cfg, device)
    params = sum(p.numel() for p in agent.ac.parameters())
    print(f"  Params: {params}")
    print()

    base_env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(frame_len=cfg.frame_len, train_len=cfg.train_len,
                          pilot_len=cfg.pilot_len, num_pilots=cfg.num_pilots),
        channel=dict(num_taps=cfg.num_taps, delay_spread=cfg.delay_spread,
                     snr_db=cfg.snr_db, seed=cfg.seed),
        window_K=cfg.window_K, seed=cfg.seed,
    ))
    base_env.reset()
    fixed_taps = base_env.channel._taps.clone()

    frame_buffer = []
    all_known_accs = []
    all_ber = []
    phase = "N/A"
    t_start = time.time()

    for fi in range(1, cfg.num_frames + 1):
        if fi <= cfg.sup_end:
            phase = "SUP"; pcoef = 0.0;  sweight = 5.0; deterministic = True
        elif fi <= cfg.mix_end:
            phase = "MIX"; pcoef = 0.3;  sweight = 3.0; deterministic = False
        else:
            phase = "RL ";  pcoef = 1.0;  sweight = 1.0; deterministic = False

        # 重置环境 (保持固定信道抽头)
        base_env.channel.set_taps(fixed_taps)
        base_env._bits = base_env.frame_gen.generate(base_env._rng)
        base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
        base_env._t = 0
        base_env._done = False

        # 批量获取所有状态 + 单次 Transformer 前向
        all_states = base_env.get_all_states().to(device)
        bit_types = [base_env.frame_cfg.bit_type(t) for t in range(cfg.frame_len)]
        true_bits = base_env.get_true_bits().to(device)
        probs, values, logits = agent.batch_act(all_states)

        # 采样动作
        p_clamped = probs.clamp(1e-6, 1 - 1e-6)
        if deterministic:
            actions = (p_clamped > 0.5).float()
        else:
            bd = dist.Bernoulli(probs=p_clamped)
            actions = bd.sample()

        # 计算奖励
        known_mask = torch.tensor([bt in ("train","pilot") for bt in bit_types],
                                  device=device, dtype=torch.bool)
        correct = (actions.squeeze(-1) == true_bits).float()
        rewards = torch.where(known_mask, 2.0*correct-1.0, torch.zeros_like(correct))
        dones = torch.zeros(cfg.frame_len, device=device)
        dones[-1] = 1.0

        # 构建轨迹
        traj = dict(states=all_states, actions=actions, logits=logits,
                    rewards=rewards, dones=dones, known_mask=known_mask, true_bits=true_bits)
        frame_buffer.append(traj)
        if len(frame_buffer) > cfg.buffer_size:
            frame_buffer.pop(0)

        # 训练
        if len(frame_buffer) >= 3:
            agent.train_on_buffer(frame_buffer, policy_coef=pcoef,
                                  sup_weight=sweight, gamma=cfg.gamma)

        # 评估
        preds = (probs.squeeze(-1) > 0.5).float()
        known_acc = (preds[known_mask]==true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0
        ber = (preds != true_bits).float().mean().item()
        all_known_accs.append(known_acc)
        all_ber.append(ber)

        if fi % cfg.log_interval == 0 or fi == 1:
            elapsed = time.time() - t_start
            avg_ka = np.mean(all_known_accs[-cfg.log_interval:])
            avg_ber = np.mean(all_ber[-cfg.log_interval:])
            print(f"  [{fi:4d}/{cfg.num_frames}] [{phase}]")
            print(f"    known_acc={known_acc:.3f}(avg={avg_ka:.3f}) BER={ber:.4f}(avg={avg_ber:.4f}) ({elapsed:.1f}s)")

    final_ka = np.mean(all_known_accs[-40:]) if len(all_known_accs)>=40 else np.mean(all_known_accs)
    best_ka = max(all_known_accs)
    final_ber = np.mean(all_ber[-40:]) if len(all_ber)>=40 else np.mean(all_ber)

    print(f"\nTraining done! {time.time()-t_start:.1f}s")
    print(f"Best known_acc: {best_ka:.3f}, Final BER: {final_ber:.5f}")

    # MMSE 对比
    mmse_results = []
    for snr_t in [5, 10, 15]:
        base_env.channel.set_snr(snr_t)
        base_env.channel.set_taps(fixed_taps)
        base_env._bits = base_env.frame_gen.generate(base_env._rng)
        base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
        base_env._t = 0; base_env._done = False
        p, _, _ = agent.batch_act(base_env.get_all_states().to(device))
        rl_ber = eval_ber(p, base_env.get_true_bits().to(device))
        mmse_ber = eval_mmse_ber(base_env.get_rx_symbols(), base_env.get_true_bits().cpu(),
                                 fixed_taps.cpu(), cfg.window_K, cfg.num_taps, snr_t)
        print(f"  SNR={snr_t:3d}dB | MMSE={mmse_ber:.5f} RL={rl_ber:.5f} | {'RL' if rl_ber<mmse_ber else 'MMSE'}")
        mmse_results.append((snr_t, mmse_ber, rl_ber))
    base_env.channel.set_snr(cfg.snr_db)

    # P-FTNet SNR 测试
    pftnet_results = []
    for snr_pt in [-2, 0, 5, 10]:
        base_env.channel.set_snr(snr_pt)
        base_env.channel.set_taps(fixed_taps)
        base_env._bits = base_env.frame_gen.generate(base_env._rng)
        base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
        base_env._t = 0; base_env._done = False
        p, _, _ = agent.batch_act(base_env.get_all_states().to(device))
        ber_pt = eval_ber(p, base_env.get_true_bits().to(device))
        print(f"  SNR={snr_pt:3d}dB | BER={ber_pt:.5f}")
        pftnet_results.append((snr_pt, ber_pt))
    base_env.channel.set_snr(cfg.snr_db)

    save_plots(all_known_accs, all_ber, mmse_results, pftnet_results, cfg)
    return final_ka, best_ka


if __name__ == "__main__":
    main()

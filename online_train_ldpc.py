# -*- coding: utf-8 -*-
"""
RL4EQ — Transformer-A2C + LDPC 在线信道均衡主程序

与 online_train.py 的区别:
  - 数据位使用 LDPC 编码: 128 位原始数据 → 256 位编码数据
  - 评估时加入 BP 解码, 测量解码后 BER
  - 对比编码前/后 BER

帧结构: | 训练(128) | 导频(64) | 数据1(128) | 导频(64) | 数据2(128) |
  其中数据1+数据2 = 256 位 = 1 个 LDPC 码字 (n=256, k=128)

三阶段训练策略:
  SUP [1-10]:  纯监督学习
  MIX [11-30]: 混合模式
  RL [31+]:    全 A2C 在线学习

用法:
  python online_train_ldpc.py                     # 200 帧, 10dB
  python online_train_ldpc.py --snr 5             # 自定义 SNR
  python online_train_ldpc.py --num_frames 50     # 自定义帧数
"""

import sys, time, os, argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig
from env.channel_models import RayleighMultipathChannel
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic import ActorCritic, TransformerConfig
from env.ldpc_coding import LDPC


@dataclass
class TrainConfig:
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
    ldpc_n: int = 256
    ldpc_k: int = 128


class A2CAgent:
    """A2C 智能体包装类。"""

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
        T = states.shape[0]
        mask = self.ac._get_causal_mask(T, self.device)
        probs, values, logits = self.ac.forward(states.unsqueeze(0), mask=mask)
        return probs.squeeze(0), values.squeeze(0), logits.squeeze(0)

    def train_on_buffer(self, frame_buffer, policy_coef, sup_weight, gamma):
        self.ac.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        c = self.cfg
        for tb in frame_buffer:
            T = tb["states"].shape[0]
            returns = torch.zeros(T, device=self.device)
            G = 0.0
            for t in reversed(range(T)):
                G = tb["rewards"][t].item() + gamma * (1 - tb["dones"][t].item()) * G
                returns[t] = G
            mask = self.ac._get_causal_mask(T, self.device)
            _, _, out_l = self.ac.forward(tb["states"].unsqueeze(0), mask=mask)
            logits = out_l.squeeze(0)
            probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            bd = dist.Bernoulli(probs=probs)
            log_probs = bd.log_prob(tb["actions"]).squeeze(0)
            values = self.ac.critic(
                self.ac.transformer(
                    self.ac.input_proj(tb["states"].unsqueeze(0))
                    + self.ac.pos_encoding(torch.arange(T, device=self.device).unsqueeze(0))
                )
            ).squeeze(-1).squeeze(0)
            adv = returns - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            known = tb["known_mask"]
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


# ---------------------------------------------------------------
# LDPC 帧生成
# ---------------------------------------------------------------
def generate_ldpc_frame(ldpc, rng, frame_cfg):
    """生成带 LDPC 编码的帧比特。

    帧结构: |训练(128)|导频1(64)|数据1(128)|导频2(64)|数据2(128)|
    数据位共 256 位 = 1 个 LDPC 码字 (n=256), 原始数据 128 位 (k=128)
    返回:
      bits: 完整帧 (frame_len,) torch.float32
      original_data: 原始 128 位 (np.int8)
      coded_data: 编码后 256 位 (np.int8)
    """
    original_data = rng.integers(0, 2, size=ldpc.k).astype(np.int8)
    coded_data = ldpc.encode(original_data)

    bits = rng.integers(0, 2, size=frame_cfg.frame_len).astype(np.float32)
    # 计算数据块位置
    pilot_starts = frame_cfg.pilot_positions
    dp1_start = pilot_starts[0] + frame_cfg.pilot_len
    dp1_end = pilot_starts[1]
    dp2_start = pilot_starts[1] + frame_cfg.pilot_len
    dp2_end = frame_cfg.frame_len

    data1_len = dp1_end - dp1_start
    data2_len = dp2_end - dp2_start

    bits[dp1_start:dp1_end] = coded_data[0:data1_len].astype(np.float32)
    bits[dp2_start:dp2_end] = coded_data[data1_len:data1_len+data2_len].astype(np.float32)

    return torch.from_numpy(bits), original_data, coded_data


@torch.no_grad()
def eval_ber(probs, true_bits):
    preds = (probs.squeeze(-1) > 0.5).float()
    return (preds != true_bits).float().mean().item()


@torch.no_grad()
def eval_mmse_ber(rx_symbols, true_bits, taps, window_K, num_taps, snr_db):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig(snr_db=args.snr, num_frames=args.num_frames, device=args.device)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # LDPC 初始化
    ldpc = LDPC(n=cfg.ldpc_n, k=cfg.ldpc_k)
    ldpc_rng = np.random.default_rng(cfg.seed + 100)

    print("=" * 65)
    print("  RL4EQ - Transformer-A2C + LDPC 在线信道均衡")
    print("=" * 65)
    print(f"  帧: {cfg.frame_len}|训练 {cfg.train_len}|导频 2x{cfg.pilot_len}|数据 {cfg.frame_len-cfg.train_len-2*cfg.pilot_len}")
    print(f"  LDPC: ({cfg.ldpc_n},{cfg.ldpc_k}) 码率 {cfg.ldpc_k/cfg.ldpc_n:.2f}")
    print(f"  信道: {cfg.num_taps}抽头 瑞利 SNR={cfg.snr_db}dB")
    print(f"  网络: Transformer {cfg.n_layers}L {cfg.n_heads}H")
    print(f"  阶段: SUP(1-{cfg.sup_end}) MIX({cfg.sup_end+1}-{cfg.mix_end}) RL({cfg.mix_end+1}+)")

    agent = A2CAgent(cfg, device)
    params = sum(p.numel() for p in agent.ac.parameters())
    print(f"  参数量: {params}")
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
    all_coded_ber = []
    all_dec_ber = []
    phase = "N/A"
    t_start = time.time()

    # 预计算数据块位置
    pilot_starts = base_env.frame_cfg.pilot_positions
    dp1_start = pilot_starts[0] + base_env.frame_cfg.pilot_len
    dp1_end = pilot_starts[1]
    dp2_start = pilot_starts[1] + base_env.frame_cfg.pilot_len
    dp2_end = base_env.frame_cfg.frame_len
    data_indices = list(range(dp1_start, dp1_end)) + list(range(dp2_start, dp2_end))
    data_tensor = torch.tensor(data_indices, dtype=torch.long)

    for fi in range(1, cfg.num_frames + 1):
        if fi <= cfg.sup_end:
            phase = "SUP"; pcoef = 0.0;  sweight = 5.0; deterministic = True
        elif fi <= cfg.mix_end:
            phase = "MIX"; pcoef = 0.3;  sweight = 3.0; deterministic = False
        else:
            phase = "RL ";  pcoef = 1.0;  sweight = 1.0; deterministic = False

        # 生成 LDPC 编码帧
        bits, orig_data, coded_data = generate_ldpc_frame(ldpc, ldpc_rng, base_env.frame_cfg)

        base_env.channel.set_taps(fixed_taps)
        base_env._bits = bits
        base_env._tx_symbols = base_env.frame_gen.modulate(bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
        base_env._t = 0; base_env._done = False

        all_states = base_env.get_all_states().to(device)
        bit_types = [base_env.frame_cfg.bit_type(t) for t in range(cfg.frame_len)]
        true_bits = base_env.get_true_bits().to(device)
        probs, values, logits = agent.batch_act(all_states)

        p_clamped = probs.clamp(1e-6, 1 - 1e-6)
        if deterministic:
            actions = (p_clamped > 0.5).float()
        else:
            bd = dist.Bernoulli(probs=p_clamped)
            actions = bd.sample()

        known_mask = torch.tensor([bt in ("train", "pilot") for bt in bit_types],
                                  device=device, dtype=torch.bool)
        correct = (actions.squeeze(-1) == true_bits).float()
        rewards = torch.where(known_mask, 2.0 * correct - 1.0, torch.zeros_like(correct))
        dones = torch.zeros(cfg.frame_len, device=device)
        dones[-1] = 1.0

        traj = dict(states=all_states, actions=actions, logits=logits,
                    rewards=rewards, dones=dones, known_mask=known_mask, true_bits=true_bits)
        frame_buffer.append(traj)
        if len(frame_buffer) > cfg.buffer_size:
            frame_buffer.pop(0)

        if len(frame_buffer) >= 3:
            agent.train_on_buffer(frame_buffer, policy_coef=pcoef,
                                  sup_weight=sweight, gamma=cfg.gamma)

        # 评估
        preds = (probs.squeeze(-1) > 0.5).float()
        known_acc = (preds[known_mask] == true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0

        # 编码后 BER (数据位)
        coded_probs = probs.squeeze(-1)[data_tensor.to(device)]
        coded_preds = (coded_probs > 0.5).float()
        coded_true = true_bits[data_tensor.to(device)]
        coded_ber = (coded_preds != coded_true).float().mean().item()

        # LDPC 解码 BER
        coded_probs_np = coded_probs.cpu().numpy()
        llr = ldpc.soft_to_llr(coded_probs_np)
        decoded = ldpc.decode(llr, max_iter=50)
        dec_ber = float(np.mean(decoded[:cfg.ldpc_k] != orig_data[:cfg.ldpc_k]))

        all_known_accs.append(known_acc)
        all_coded_ber.append(coded_ber)
        all_dec_ber.append(dec_ber)

        if fi % cfg.log_interval == 0 or fi == 1:
            elapsed = time.time() - t_start
            avg_ka = np.mean(all_known_accs[-cfg.log_interval:])
            avg_cber = np.mean(all_coded_ber[-cfg.log_interval:])
            avg_dber = np.mean(all_dec_ber[-cfg.log_interval:])
            print(f"  [{fi:4d}/{cfg.num_frames}] [{phase}]")
            print(f"    known_acc={known_acc:.3f}(avg={avg_ka:.3f}) "
                  f"coded_BER={coded_ber:.4f}(avg={avg_cber:.4f}) "
                  f"dec_BER={dec_ber:.4f}(avg={avg_dber:.4f}) ({elapsed:.1f}s)")

    final_ka = np.mean(all_known_accs[-40:])
    best_ka = max(all_known_accs)
    final_cber = np.mean(all_coded_ber[-40:])
    final_dber = np.mean(all_dec_ber[-40:])

    print(f"\n训练完成! {time.time()-t_start:.1f}s")
    print(f"  Best known_acc: {best_ka:.3f}")
    print(f"  Final coded_BER: {final_cber:.5f}")
    print(f"  Final decoded_BER (LDPC): {final_dber:.5f}")

    # MMSE + LDPC 对比
    print("\n--- MMSE + LDPC 对比 ---")
    for snr_t in [5, 10, 15]:
        ldpc_rng2 = np.random.default_rng(cfg.seed + snr_t)
        bits_t, orig_t, coded_t = generate_ldpc_frame(ldpc, ldpc_rng2, base_env.frame_cfg)

        base_env.channel.set_snr(snr_t)
        base_env.channel.set_taps(fixed_taps)
        base_env._bits = bits_t
        base_env._tx_symbols = base_env.frame_gen.modulate(bits_t)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
        base_env._t = 0; base_env._done = False

        p_t, _, _ = agent.batch_act(base_env.get_all_states().to(device))
        rl_ber = eval_ber(p_t, base_env.get_true_bits().to(device))

        mmse_ber = eval_mmse_ber(base_env.get_rx_symbols(), base_env.get_true_bits().cpu(),
                                 fixed_taps.cpu(), cfg.window_K, cfg.num_taps, snr_t)

        coded_p_t = p_t.squeeze(-1)[data_tensor.to(device)]
        llr_t = ldpc.soft_to_llr(coded_p_t.cpu().numpy())
        decoded_t = ldpc.decode(llr_t, max_iter=50)
        ldpc_ber_t = float(np.mean(decoded_t[:cfg.ldpc_k] != orig_t[:cfg.ldpc_k]))

        print(f"  SNR={snr_t:3d}dB | MMSE={mmse_ber:.5f} RL={rl_ber:.5f} RL+LDPC={ldpc_ber_t:.5f}")
    base_env.channel.set_snr(cfg.snr_db)


if __name__ == "__main__":
    main()

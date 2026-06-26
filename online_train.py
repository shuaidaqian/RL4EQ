# -*- coding: utf-8 -*-
"""
RL4EQ — PPO 在线信道均衡主程序

算法: PPO + GAE (替代原 A2C-KPG)
网络: Transformer 编码器 Actor-Critic
信道: P-FTNet 参数 (16 抽头频率选择性瑞利衰落)

核心改进 (相比 A2C):
  1. GAE (λ=0.95) — 更有效传播稀疏导频奖励
  2. PPO-Clip (ε=0.2) — 限制策略更新步长，防止 burst error 破坏
  3. 优势归一化 — 平衡导频/数据段梯度量级
  4. KL 早停 — 策略变化过快时提前停止
  5. 多 epoch 数据复用 (K=4) — 每帧数据利用更充分

用法:
  python online_train.py                     # 默认 200 帧, 10dB, PPO
  python online_train.py --snr 5             # 自定义 SNR
  python online_train.py --num_frames 200    # 自定义帧数
  python online_train.py --algo ppo          # PPO 算法
  python online_train.py --algo a2c          # 对比 A2C
  python online_train.py --ldpc              # 数据段使用 LDPC(256,128) 编码
"""

import sys, time, os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.frame_structure import FrameConfig
from env.rayleigh_channel import RayleighMultipathChannel
from env.comm_env import CommunicationEnv, EnvConfig
from agent.actor_critic import ActorCritic, TransformerConfig
from agent.ppo import PPOTrainer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─── LDPC 支持 ─────────────────────────────────────

def _maybe_import_ldpc():
    """延迟导入 LDPC 模块（仅 --ldpc 时加载）。"""
    try:
        from env.ldpc_coding import LDPC
        return LDPC
    except ImportError:
        return None


# ─── 配置 ────────────────────────────────────────────

class TrainConfig:
    """训练参数。"""
    def __init__(self, args):
        self.num_frames = args.num_frames
        self.log_interval = 10
        self.seed = 42
        self.device = args.device
        self.snr_db = args.snr
        self.algo = args.algo.lower()

        # 帧结构
        self.frame_len = 512
        self.train_len = 128
        self.pilot_len = 64
        self.num_pilots = 2

        # 信道
        self.num_taps = 16
        self.delay_spread = 10
        self.window_K = 10
        self.state_dim = 45

        # 网络
        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 3
        self.dim_feedforward = 256

        # RL 超参
        self.gamma = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.actor_lr = 3e-4
        self.buffer_size = 5

        # PPO 专用
        self.lam = 0.95
        self.clip_eps = 0.2
        self.k_epochs = 8
        self.kl_threshold_init = 0.05  # 初始KL阈值（宽松）
        self.kl_threshold_final = 0.01 # 最终KL阈值（收紧）

        # A2C 专用
        self.sup_end = 10
        self.mix_end = 30

        # 训练模式
        self.sup_weight = 1.0
        self.use_ldpc = getattr(args, 'ldpc', False)

        # LDPC 参数（仅 --ldpc 时使用）
        self.ldpc_n = 256
        self.ldpc_k = 128


# ─── 智能体 ──────────────────────────────────────────

class Agent:
    """统一智能体接口，支持 A2C 和 PPO 两种更新方式。"""

    def __init__(self, cfg: TrainConfig, device):
        self.cfg = cfg
        self.device = device
        self.algo = cfg.algo

        self.tf_cfg = TransformerConfig(
            state_dim=cfg.state_dim, d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            dim_feedforward=cfg.dim_feedforward,
        )
        self.ac = ActorCritic(self.tf_cfg).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.actor_lr)

        # PPO 训练器 (仅 ppo 模式使用)
        # finetune 模式: 降低学习率和epoch, 收紧KL
        self._is_finetune = getattr(self.cfg, "_finetune_mode", False)
        if self._is_finetune:
            self.cfg.actor_lr = 1e-4
            self.cfg.k_epochs = 4
            self.cfg.kl_threshold_init = 0.02
            self.cfg.kl_threshold_final = 0.005
            self.cfg.entropy_coef = 0.005
            self._finetune_unfreeze_step = 200  # 前200帧冻结编码器




        self.ppo_trainer = PPOTrainer(
            self.ac, lr=cfg.actor_lr, gamma=cfg.gamma, lam=cfg.lam,
            clip_eps=cfg.clip_eps, value_coef=cfg.value_coef,
            ent_coef=cfg.entropy_coef, max_grad_norm=cfg.max_grad_norm,
        ) if cfg.algo == 'ppo' else None

        # 帧缓冲 (a2c 模式使用)
        self.frame_buffer = []

    @torch.no_grad()
    def batch_act(self, states):
        """批量整帧前向推理。"""
        T = states.shape[0]
        mask = self.ac._get_causal_mask(T, self.device)
        probs, values, logits = self.ac.forward(states.unsqueeze(0), mask=mask)
        return probs.squeeze(0), values.squeeze(0), logits.squeeze(0)

    def train_a2c(self, tb, policy_coef, sup_weight):
        """A2C-KPG 多帧缓冲更新。"""
        self.ac.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        c = self.cfg

        for frame in tb:
            T = frame["states"].shape[0]

            # MC 回报
            returns = torch.zeros(T, device=self.device)
            G = 0.0
            for t_idx in reversed(range(T)):
                G = frame["rewards"][t_idx].item() + c.gamma * (1 - frame["dones"][t_idx].item()) * G
                returns[t_idx] = G

            # Transformer 前向
            mask = self.ac._get_causal_mask(T, self.device)
            _, _, out_l = self.ac.forward(frame["states"].unsqueeze(0), mask=mask)
            logits = out_l.squeeze(0)

            probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            bd = dist.Bernoulli(probs=probs)
            log_probs = bd.log_prob(frame["actions"]).squeeze(0)

            # 价值
            values = self.ac.critic(
                self.ac.transformer(
                    self.ac.input_proj(frame["states"].unsqueeze(0))
                    + self.ac.pos_encoding(torch.arange(T, device=self.device).unsqueeze(0))
                )
            ).squeeze(-1).squeeze(0)

            adv = returns - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            known = frame["known_mask"]

            if known.any():
                p_loss = -(log_probs[known] * adv[known]).mean()
                s_loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1)[known], frame["true_bits"][known])
            else:
                p_loss = torch.zeros(1, device=self.device)
                s_loss = torch.zeros(1, device=self.device)

            v_loss = F.mse_loss(values, returns)
            e_loss = bd.entropy().squeeze(0).mean()

            total_loss += (policy_coef * p_loss + c.value_coef * v_loss
                           - c.entropy_coef * e_loss + sup_weight * s_loss)

        total_loss = total_loss / len(tb)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
        self.optimizer.step()

    def train_ppo(self, tb, kl_threshold):
        """PPO + GAE 更新 (使用帧缓冲中所有帧)。"""
        # finetune 模式: 前 N 帧冻结编码器, 只调 Actor/Critic 头
        if self._is_finetune:
            frame_idx = getattr(self.cfg, "_current_frame", 0)
            for name, param in self.ac.named_parameters():
                if "actor_fc" in name or "critic" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = (frame_idx > self._finetune_unfreeze_step)
        for frame in tb:
            self.ppo_trainer.train_on_trajectory(
                states=frame["states"],
                actions=frame["actions"],
                log_probs=frame["log_probs"],
                rewards=frame["rewards"],
                values=frame["values"],
                dones=frame["dones"],
                k_epochs=self.cfg.k_epochs,
                kl_threshold=kl_threshold,
            )


# ─── 评估 ──────────────────────────────────────────

@torch.no_grad()
def eval_ber(probs, true_bits):
    preds = (probs.squeeze(-1) > 0.5).float()
    return (preds != true_bits).float().mean().item()


# ─── 基线对比与绘图 ────────────────────────────────

def eval_and_plot(agent, cfg, device, base_env, fixed_taps):
    """训练完成后执行基线对比评估并生成可视化图像。"""
    from utils.baselines import run_baselines

    snr_points = [0, 2, 5, 8, 10, 12, 15]
    results = {"SNR": [], "RL": [], "MMSE": [], "DFE": []}

    print("\n" + "=" * 65)
    print("  基线对比评估 (BER vs SNR)")
    print("=" * 65)

    for snr in snr_points:
        base_env.set_snr(snr)
        base_env.channel.set_taps(fixed_taps)
        base_env._bits = base_env.frame_gen.generate(base_env._rng)
        base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)

        states = base_env.get_all_states().to(device)
        probs, _, _ = agent.batch_act(states)
        true_bits = base_env.get_true_bits().to(device)
        rl_ber = eval_ber(probs, true_bits)

        baselines = run_baselines(
            base_env.get_rx_symbols(), base_env.get_true_bits().cpu(),
            fixed_taps.cpu(), cfg.window_K, cfg.num_taps, snr,
        )

        results["SNR"].append(snr)
        results["RL"].append(rl_ber)
        results["MMSE"].append(baselines["MMSE"])
        results["DFE"].append(baselines["DFE"])
        print(f"  SNR={snr:3d}dB | RL={rl_ber:.5f} | MMSE={baselines['MMSE']:.5f} | DFE={baselines['DFE']:.5f}")

    base_env.set_snr(cfg.snr_db)

    if HAS_MPL:
        save_dir = "logs"
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        colors = {"RL": "#1565C0", "MMSE": "#E53935", "DFE": "#F9A825"}
        markers = {"RL": "o-", "MMSE": "s--", "DFE": "d-."}
        for method in ["MMSE", "DFE", "RL"]:
            ax.semilogy(results["SNR"], results[method], markers[method],
                        color=colors[method], label=method, linewidth=1.5, markersize=6)
        ax.axhline(0.01, color="gray", ls=":", alpha=0.5, label="BER=0.01")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title("BER vs SNR Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-3, 0.6)

        ax = axes[1]
        gains_mmse = [np.log10(b / r) if r > 1e-6 else 3.0 for b, r in zip(results["MMSE"], results["RL"])]
        gains_dfe = [np.log10(b / r) if r > 1e-6 else 3.0 for b, r in zip(results["DFE"], results["RL"])]
        x = np.arange(len(snr_points))
        w = 0.3
        ax.bar(x - w / 2, gains_mmse, w, label="RL vs MMSE", color=colors["MMSE"], alpha=0.7)
        ax.bar(x + w / 2, gains_dfe, w, label="RL vs DFE", color=colors["DFE"], alpha=0.7)
        ax.axhline(0, color="gray", ls="-", lw=0.5)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER Reduction (log10)")
        ax.set_title("RL Improvement over Baselines")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in snr_points])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "baseline_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"  Baseline comparison saved to {save_dir}/baseline_comparison.png")
        plt.close()

    return results


# ─── 可视化 ──────────────────────────────────────────

def save_training_plots(all_known_accs, all_ber, cfg, save_dir="logs"):
    """保存训练曲线（known_acc 和 BER）到 PNG。"""
    if not HAS_MPL:
        print("  Warning: matplotlib not available, skip plotting")
        return
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    frames = np.arange(1, len(all_known_accs) + 1)

    def smooth(data, w=11):
        return np.convolve(data, np.ones(w) / w, mode="valid") if len(data) >= w else data

    ax1.plot(frames, all_known_accs, alpha=0.3, color="#2196F3", lw=0.8, label="Raw")
    if len(all_known_accs) >= 11:
        sk = smooth(all_known_accs, 11)
        ax1.plot(np.arange(11, len(all_known_accs) + 1), sk, color="#1565C0", lw=2, label="Smooth(w=11)")
    ax1.axhline(0.9, color="#4CAF50", ls="--", alpha=0.5, label="Known-Acc=0.9")
    ax1.set_ylabel("Known-Acc")
    ax1.set_title(f"Training Curve ({cfg.algo.upper()}, SNR={cfg.snr_db}dB)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(frames, all_ber, alpha=0.3, color="#F44336", lw=0.8, label="Raw")
    if len(all_ber) >= 11:
        sb = smooth(all_ber, 11)
        ax2.plot(np.arange(11, len(all_ber) + 1), sb, color="#C62828", lw=2, label="Smooth(w=11)")
    ax2.axhline(0.01, color="#4CAF50", ls="--", alpha=0.5, label="BER=0.01")
    ax2.axhline(0.5, color="#FF9800", ls=":", alpha=0.5, label="Random")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("BER")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 0.6)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curve saved to {save_dir}/training_curve.png")

    # BER 放大版
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, all_ber, color="#1565C0", lw=1, label=f"{cfg.algo.upper()} BER")
    ax.axhline(0.01, color="#4CAF50", ls="--", lw=1, alpha=0.7, label="Target (BER=0.01)")
    ax.axhline(0.5, color="#FF9800", ls=":", lw=1, alpha=0.5, label="Random (BER=0.5)")
    best_idx = np.argmin(all_ber)
    ax.scatter(best_idx + 1, all_ber[best_idx], color="#E53935", s=50, zorder=5,
               label=f"Best={all_ber[best_idx]:.5f}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("BER")
    ax.set_title(f"BER Progress (SNR={cfg.snr_db}dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ber_progress.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  BER progress saved to {save_dir}/ber_progress.png")


# ─── 主程序 ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--algo", type=str, default="ppo", choices=["a2c", "ppo"])
    parser.add_argument("--ldpc", action="store_true", help="数据段使用 LDPC(256,128) 编码")
    parser.add_argument("--eval", action="store_true", help="训练完成后执行基线对比评估并绘图")
    parser.add_argument("--finetune", type=str, default=None,
                        help="加载预训练权重路径")
    args = parser.parse_args()

    cfg = TrainConfig(args)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print(f"  RL4EQ — {'PPO+GAE' if cfg.algo=='ppo' else 'A2C'} Online Channel Equalization")
    print("=" * 72)
    print(f"  Frame: 512 | train 128 | pilot 2x64 | data {512-128-128}")
    print(f"  Channel: {cfg.num_taps}-tap Rayleigh, SNR={cfg.snr_db}dB")
    print(f"  Network: Transformer {cfg.n_layers}L {cfg.n_heads}H")
    if cfg.algo == 'ppo':
        print(f"  PPO: GAE λ={cfg.lam} clip_ε={cfg.clip_eps} K_epochs={cfg.k_epochs}")
    else:
        print(f"  A2C-KPG: buffer_size={cfg.buffer_size} SUP(1-{cfg.sup_end}) MIX({cfg.sup_end+1}-{cfg.mix_end})")
    print()

    agent = Agent(cfg, device)
    params = sum(p.numel() for p in agent.ac.parameters())
    print(f"  Params: {params}")
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location=device, weights_only=True)
        agent.ac.load_state_dict(ckpt, strict=False)
        print(f"  Loaded pretrained weights from {args.finetune}")
        cfg._finetune_mode = True
        if cfg.num_frames == 200:
            cfg.num_frames = 300

    # 环境
    base_env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(frame_len=cfg.frame_len, train_len=cfg.train_len,
                          pilot_len=cfg.pilot_len, num_pilots=cfg.num_pilots),
        channel=dict(num_taps=cfg.num_taps, delay_spread=cfg.delay_spread,
                     snr_db=cfg.snr_db, seed=cfg.seed),
        window_K=cfg.window_K, seed=cfg.seed,
    ))
    base_env.reset()
    fixed_taps = base_env.channel._taps.clone()

    # LDPC 初始化（如果启用）
    ldpc = None
    all_dec_ber = None
    if cfg.use_ldpc:
        LDPC = _maybe_import_ldpc()
        if LDPC is None:
            print("  WARNING: LDPC 模块未找到，跳过 LDPC 编码")
            cfg.use_ldpc = False
        else:
            ldpc = LDPC(n=cfg.ldpc_n, k=cfg.ldpc_k)
            all_dec_ber = []
            pilot_starts = base_env.frame_cfg.pilot_positions
            dp1_start = pilot_starts[0] + base_env.frame_cfg.pilot_len
            dp1_end = pilot_starts[1]
            dp2_start = pilot_starts[1] + base_env.frame_cfg.pilot_len
            dp2_end = base_env.frame_cfg.frame_len
            data_indices = list(range(dp1_start, dp1_end)) + list(range(dp2_start, dp2_end))
            data_tensor = torch.tensor(data_indices, dtype=torch.long)
            ldpc_rng = np.random.default_rng(cfg.seed + 100)
            print(f"  LDPC: ({cfg.ldpc_n},{cfg.ldpc_k})")
    print()

    all_known_accs = []
    all_ber = []
    t_start = time.time()
    min_ber = 1.0
    best_ber = 1.0
    low_ber_frames = 0

    for fi in range(1, cfg.num_frames + 1):
        cfg._current_frame = fi
        # 阶段调度 (仅 A2C 模式)
        if cfg.algo == 'a2c':
            if fi <= cfg.sup_end:
                phase = "SUP"; pcoef = 0.0;  sweight = 5.0; det = True
            elif fi <= cfg.mix_end:
                phase = "MIX"; pcoef = 0.3;  sweight = 3.0; det = False
            else:
                phase = "RL ";  pcoef = 1.0;  sweight = 1.0; det = False
        else:
            phase = "PPO"; pcoef = 1.0; sweight = 0.0; det = False

        # KL 阈值线性退火：初始宽松 → 后期收紧
        kl_ratio = (fi - 1) / max(cfg.num_frames - 1, 1)
        kl_threshold = (cfg.kl_threshold_init - cfg.kl_threshold_final) * (1.0 - kl_ratio) + cfg.kl_threshold_final

        # 重置环境
        base_env.channel.set_taps(fixed_taps)

        if cfg.use_ldpc and ldpc is not None:
            # LDPC 编码帧
            original_data = ldpc_rng.integers(0, 2, size=ldpc.k).astype(np.int8)
            coded_data = ldpc.encode(original_data)
            bits = np.random.default_rng().integers(0, 2, size=cfg.frame_len).astype(np.float32)
            pilot_starts = base_env.frame_cfg.pilot_positions
            dp1_start = pilot_starts[0] + cfg.pilot_len;
            dp1_end = pilot_starts[1]
            dp2_start = pilot_starts[1] + cfg.pilot_len;
            dp2_end = cfg.frame_len
            bits[dp1_start:dp1_end] = coded_data[0:dp1_end-dp1_start].astype(np.float32)
            bits[dp2_start:dp2_end] = coded_data[dp1_end-dp1_start:dp1_end-dp1_start+dp2_end-dp2_start].astype(np.float32)
            base_env._bits = torch.from_numpy(bits)
        else:
            base_env._bits = base_env.frame_gen.generate(base_env._rng)

        base_env._tx_symbols = base_env.frame_gen.modulate(base_env._bits)
        base_env._rx_symbols = base_env.channel.convolve(base_env._tx_symbols)
        base_env._rx_symbols = base_env.channel.add_awgn(base_env._rx_symbols)
        base_env._t = 0
        base_env._done = False

        # 批量状态 + 前向
        all_states = base_env.get_all_states().to(device)
        bit_types = [base_env.frame_cfg.bit_type(t) for t in range(cfg.frame_len)]
        true_bits = base_env.get_true_bits().to(device)
        probs, values, logits = agent.batch_act(all_states)

        # 采样动作
        p_clamped = probs.clamp(1e-6, 1 - 1e-6)
        bd = dist.Bernoulli(probs=p_clamped)
        actions = bd.sample()

        # 计算奖励和 mask
        known_mask = torch.tensor([bt in ("train", "pilot") for bt in bit_types],
                                  device=device, dtype=torch.bool)
        correct = (actions.squeeze(-1) == true_bits).float()
        rewards = torch.where(known_mask, 2.0 * correct - 1.0, torch.zeros_like(correct))
        if getattr(cfg, "_finetune_mode", False):
            dpen = -0.2 * (p_clamped.squeeze(-1) - 0.5).abs()
            rewards = torch.where(known_mask, rewards, dpen)
        dones = torch.zeros(cfg.frame_len, device=device)
        dones[-1] = 1.0

        # 算 log_probs 和 values (PPO 模式需要存储)
        log_probs = bd.log_prob(actions)

        # 构建轨迹
        traj = dict(
            states=all_states, actions=actions, log_probs=log_probs,
            rewards=rewards, values=values.squeeze(-1),
            dones=dones, known_mask=known_mask, true_bits=true_bits,
        )

        # === 训练 ===
        if cfg.algo == 'a2c':
            agent.frame_buffer.append(traj)
            if len(agent.frame_buffer) > cfg.buffer_size:
                agent.frame_buffer.pop(0)
            if len(agent.frame_buffer) >= 3:
                agent.train_a2c(agent.frame_buffer, pcoef, sweight)
        else:
            # PPO: 单帧训练（PPO 的 ratio 机制要求旧数据来自当前策略）
            agent.train_ppo([traj], kl_threshold)

        # 评估
        preds = (probs.squeeze(-1) > 0.5).float()
        known_acc = (preds[known_mask] == true_bits[known_mask]).float().mean().item() if known_mask.any() else 0.0
        ber = (preds != true_bits).float().mean().item()
        all_known_accs.append(known_acc)
        all_ber.append(ber)

        # LDPC 解码 BER（如果启用）
        if cfg.use_ldpc and ldpc is not None:
            coded_probs = probs.squeeze(-1)[data_tensor.to(device)]
            coded_preds = (coded_probs > 0.5).float()
            coded_true = true_bits[data_tensor.to(device)]
            coded_ber = (coded_preds != coded_true).float().mean().item()
            llr = ldpc.soft_to_llr(coded_probs.cpu().numpy())
            decoded = ldpc.decode(llr, max_iter=50)
            dec_ber = float(np.mean(decoded[:cfg.ldpc_k] != original_data[:cfg.ldpc_k]))
            all_dec_ber.append(dec_ber)

        # 追踪最佳 BER 和持续低 BER 帧数
        if ber < best_ber:
            best_ber = ber
        if ber < 0.01:
            low_ber_frames += 1
        else:
            low_ber_frames = 0

        if fi % cfg.log_interval == 0 or fi == 1:
            elapsed = time.time() - t_start
            avg_ka = np.mean(all_known_accs[-cfg.log_interval:])
            avg_ber = np.mean(all_ber[-cfg.log_interval:])
            msg = (f"  [{fi:4d}/{cfg.num_frames}] [{phase}]"
                   f" known_acc={known_acc:.3f}(avg={avg_ka:.3f})"
                   f" BER={ber:.5f}(avg={avg_ber:.5f})"
                   f" best={best_ber:.5f}"
                   f" ({elapsed:.1f}s)")
            if cfg.use_ldpc and ldpc is not None and all_dec_ber:
                msg += f" dec_BER={np.mean(all_dec_ber[-cfg.log_interval:]):.4f}"

        # 目标达成: BER < 0.01 持续 10 帧
        if low_ber_frames >= 10:
            print(f"\n✅ 目标达成! BER < 0.01 持续 {low_ber_frames} 帧 (第 {fi-9}-{fi} 帧)")
            break

    elapsed = time.time() - t_start
    final_ber = np.mean(all_ber[-20:]) if len(all_ber) >= 20 else np.mean(all_ber)
    final_ka = np.mean(all_known_accs[-20:]) if len(all_known_accs) >= 20 else np.mean(all_known_accs)

    print(f"\n{'=' * 65}")
    print(f"  训练完成! ({elapsed:.1f}s) 共 {len(all_ber)} 帧")
    print(f"  算法: {cfg.algo.upper()}")
    print(f"  Best BER: {best_ber:.5f}")
    print(f"  Final avg BER (last 20): {final_ber:.5f}")
    print(f"  Final avg Known-Acc (last 20): {final_ka:.3f}")
    if cfg.use_ldpc and ldpc is not None and all_dec_ber:
        final_dber = np.mean(all_dec_ber[-20:])
        print(f"  Final avg decoded_BER (LDPC, last 20): {final_dber:.5f}")
    ok_str = "OK" if best_ber < 0.01 else "FAIL"
    print(f"  BER < 0.01 target: [{ok_str}] (best={best_ber:.5f})")
    print(f"{'=' * 65}")

    # 保存训练曲线
    save_training_plots(all_known_accs, all_ber, cfg)

    # 基线对比评估
    if hasattr(args, 'eval') and args.eval:
        eval_and_plot(agent, cfg, device, base_env, fixed_taps)

    return final_ka, best_ber


if __name__ == "__main__":
    main()

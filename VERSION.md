## v1.1-format-fix (2026-06-26)

### 修复
- 去除 pretrain.py 的 UTF-8 BOM 头 (\xef\xbb\xbf)
- agent/ppo.py 换行符统一为 CRLF (原LF)
- 创建 .gitattributes 统一git换行符处理
- 清理所有生成的中转临时文件

### 新增
- agent/actor_critic_v2.py: V2网络 (FiLM信道条件自适应层, 115K参数)
- pretrain_v8.py: V8多信道混合预训练 (阶段式: 固定信道300步 → 随机信道)
- online_train_v8.py: V8 PPO微调 (低学习率1e-4, 紧KL约束, gamma=0.99)

### 实验结论
- V8多信道混合预训练在因果掩码下不收敛 (BER≈0.5)
- V6固定信道预训练 + PPO微调仍是最优方案 (BER<0.01)
- V2网络的FiLM层结构正确，但信道条件嵌入在因果掩码下增益有限
# RL4EQ 版本记录

## v1.0-pretrain-ppofinetune (2026-06-26)

当前分支: feat/pretrain-finetune
commit base: f6db813

### 变更内容
- pretrain.py V6: 固定信道因果预训练
- online_train.py: PPO finetune 模式优化
- online_train.py: 大网络配置 (486K)
- env/comm_env.py: 45维状态 (CE实验回退)
- env/ldpc_coding.py: 修复除零bug

### 实验结果
- 固定信道预训练 best BER=0.0097
- PPO微调 SNR=10dB best BER=0.0039
- SNR=5dB best BER=0.0137
- SNRLDPC=5dB best BER=0.0098

### 实验结论
方案A(元学习)在新信道BER=0.107但微调受限
方案B(信道估计LS)在低SNR不准确
当前最佳: 固定信道预训练 + PPO微调

### 回滚方式
git checkout f6db813
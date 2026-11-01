# DFE 已知 Pilot 反馈核查：48 个配对帧的 Data BER 不变

本轮完成了与待确认的神经结构、正式输入边界无关的传统基线核查。
在固定参考 DFE 参数下，强制使用已知 Adapt 符号进行检测反馈，没有改变当前
48 个配对帧的 Data BER。该疑点未解释现有主结果差距，不扩大同设置矩阵。
不修改正式默认基线，也不将本轮结果当作神经在线收益证据。

## 检查依据与隔离实现

现有 `_dfe_rls` 在拟合权重时使用已知 Adapt 反馈；在最终检测时又从头硬判决
所有位置，包括已知的 Adapt。传统接收机可以直接在已知位置反馈已知符号，
避免重复判决该位置。这是一项合法接收规则，但改善幅度不能仅由规则合理推出。

新增 `known_pilot_feedback=False` 可选参数。显式开启时仅覆盖 Adapt 位置，
Reward 和 Data 仍由接收机判决。训练目标、权重更新和默认检测路径保持原规则。
没有使用 Reward 标签、未知发送符号、Data 标签、神经网络或 RL。

独立双精度加权岭回归参照覆盖两种检测规则、三个遗忘因子、三个先验尺度；
另测试未知标签/发送符号污染不改变接收输出，以及端到端波形配对。
测试构造首次运行出现变量定义顺序错误，修正后定向测试 **44 passed，7.14 秒**。
该错误未影响实验输出，真实实验在修正且定向测试通过后才启动。

## 固定预算与结果

[实验前计划](dfe_known_feedback_plan_20261101.md)固定 Level B、prefix、四主 SNR，
offline_train seed=43000–43002，各 4 帧；仅使用原参考 41+5 taps、遗忘因子
.995、初始逆协方差 10，不做网格选择。每种规则分别维护自己的历史 tail，
继续采用 acquisition CFO 与当前 Adapt phase 补偿。

48 个上下文、两规则，共 96 条记录。配对核验了相同接收波形和 Adapt 更新次数，
并确认 **48 对逐帧 Data BER 全部相同**。仅有 BER 记录，不能将此表述成
所有 Data 判决或全部接收输出逐 bit 相同。

| SNR | 原规则平均 Data BER | 已知 Adapt 反馈平均 Data BER | 改善百分点［seed 级 95% 区间］ |
|---:|---:|---:|---:|
| 0 dB | 21.59288% | 21.59288% | 0［0，0］ |
| 5 dB | 10.44922% | 10.44922% | 0［0，0］ |
| 10 dB | 3.52648% | 3.52648% | 0［0，0］ |
| 15 dB | 1.52995% | 1.52995% | 0［0，0］ |

统计单位为 3 个独立 seed 的帧均值，96 行不是 96 个独立样本。零宽区间只
表示本批样本没有 BER 变化，不证明所有信道、预算或参数下均无影响。
前缀中 Adapt 与 Data 之间还存在 32 个 Reward 位置，而反馈长度为 5；它们
仍需自行判决，所以已知 Adapt 反馈不直接覆盖 Data 边界的最近反馈符号。
递归差异仍可能传播，不能仅由这个间隔断言后续输出必然相同。

原始文件为被 Git 忽略的 `logs/dfe_known_feedback_3s4f_20261101.json`；来源 SHA256、
配置与精简统计见[机器摘要](results/dfe_known_feedback_stage_20261101.json)。
本轮没有神经结果，不能与其他 seed 的神经数值组成配对优势。

## 复现与后续

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/diagnose_dfe_known_feedback.py --config configs/eme_long_memory_v2_phase_transfer.json --seed-start 43000 --num-seeds 3 --frames 4 --output logs/dfe_known_feedback_3s4f_20261101.json
```

最终版本全量回归 **724 passed，248.85 秒**，使用 CPU、FFT 后端及单线程环境。
定向测试与全量回归有重叠，不相加。下一步仍应解决神经候选的可迁移性及 Reward 的可分辨性，
不继续靠扩大传统参数网格或 PPO 预算推进。正式接收范围和调制结构的两项选择
尚待答复，未将持续 goal 的自动续执行当作对这两项选择的同意。

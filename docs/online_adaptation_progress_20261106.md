# 在线微调阶段进展（2026-11-06）

## 研究主张与当前结论

本阶段沿用冻结的 Level B `eme_long_memory_v2` 长回波配置：最大时延 116 符号、Pilot=256（Adapt=224、Reward=32）、0/5/10/15 dB。目标仍是离线学习可由合法 Pilot 适配的神经均衡器，并在线真实更新神经参数；只有独立 session 上 Online 相对同 checkpoint、相同物理状态跟踪协议的 Frozen 有可重复 Data BER 改善，且预先定义的时间窗支持总体增益扩大，才算支持研究主张。

当前离线模型相对所列传统 DFE-RLS/LMMSE-FIR 的强基线优势已在[离线训练与基线报告](offline_equalizer_training_and_baseline_20261105.md)归档。它建立的是当前开发矩阵内的离线优势，不自动证明在线增益。

此前 rank-1 Fisher-GN 机制在两个独立复核 session 上没有获得整体 Online−Frozen 正收益，接受更新中也有事后 Data 退化。因此不再扩大同一 rank-1 机制矩阵；失败结果保留为机制筛查证据，不推断一般在线微调不可行。详细数值见[rank-1复核结果](results/rank1_gn_meta_fisher_eval32_rep_v1_20261106/README.md)。

## 已完成：全 head trajectory 端到端 smoke

新增配置 `configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json`，冻结来源强离线 checkpoint 后只允许全 head 在线/外层更新；离线 trajectory 训练展开连续4帧的同一会话在线过程，内循环使用当前帧 Adapt-only BCE、逐帧 Adam、独立 Reward Pilot 硬门控、参数跨帧保留及固定总半径。训练端关闭 Adam 状态跨帧持久化，历史长度为1，和当前评估器逐帧重建 Adam 的行为匹配。离线外层可以使用这些训练 episode 的 Data 标签优化多帧目标；部署更新与 Reward 门控不读取 Data 标签。

采用同一来源 checkpoint、同一训练配置和训练 seed，分别以 `--transfer-objective trajectory` 与 `--transfer-objective baseline` 跑2步烟雾测试。两臂均成功保存权重；随后在相同未见开发 session 76110、全部四个 SNR 上运行8帧 Frozen/Online 配对评估。结果文件分别位于：

- `results/full_head_trajectory_smoke_v1_20261106/eval.json`
- `results/full_head_supervised_smoke_v1_20261106/eval.json`

| SNR | Trajectory Online−Frozen收益（Frozen−Online BER） | 普通续训收益 | Trajectory晚窗−早窗 | 普通续训晚窗−早窗 | Trajectory接受/8 | 普通续训接受/8 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | +6.51e-4 | +4.88e-4 | +1.30e-3 | -1.30e-3 | 8 | 8 |
| 5 dB | -1.63e-4 | -1.63e-4 | -6.51e-4 | -6.51e-4 | 4 | 4 |
| 10 dB | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 dB | 0 | 0 | 0 | 0 | 0 | 0 |

**这些是单 session、每 SNR 8 帧的流程 smoke，不能用作增益证据或模型选择依据。** 0 dB两臂都出现正点估计，5 dB都为负，10/15 dB没有候选通过门控；数据量不足以判断 trajectory 是否优于普通续训，也不能证明随帧数增加的收益趋势。Smoke 的用途仅为确认训练、checkpoint载入、四点推理、在线更新、Reward守门与逐帧结果归档可以端到端运行。

## 已完成：matched A/B 开发筛查与决策

8步训练 A/B 与新 session 76120、76121 的16帧四 SNR 评估已完成，详细结果见[全 head trajectory 开发筛查](results/full_head_trajectory_dev8_v1_20261106/README.md)。Trajectory的四 SNR平均 Online−Frozen 收益依次为 `-1.63e-4/+1.22e-4/-4.07e-5/0`，普通监督续训为 `+2.44e-4/-4.07e-5/0/0`；trajectory相对普通续训的增益差没有跨SNR稳定为正。0 dB两个trajectory session的整体收益均为负；5 dB正点估计很小，且两会话CI很宽。早晚窗差为0 dB负、5 dB略正、高SNR为0，不能证明收益随帧推进扩大。共接受47/128个候选，5次trajectory接受后的Data BER事后恶化；这些值只作开发机制诊断。

因此不扩大当前4帧趋势损失配置，也不进入正式确认。随后对两个已训练 checkpoint 做了逐帧单步事后梯度诊断，细节见[全 head 梯度方向诊断](results/full_head_trajectory_dev8_v1_20261106/head_gradient_alignment_diagnostic.md)：0/5 dB Adapt/Data余弦约0.75/0.65且候选连续Data BCE下降，说明更新方向并非完全错误；但0 dB存在Reward接受后的Data有害候选，BER收益不稳定。10/15 dB余弦接近零、更新极少且Data变化很小，说明高SNR缺少有效更新信号。诊断每点仅2个session，且逐帧候选从起点重建，没有复现跨帧参数累计，所以只用于机制筛查。

之后已为评估轨迹加入动作后的Data BCE（仅事后评分），并在相同checkpoint和session上重跑。0/5 dB的BCE总体收益约为8.0e-4/2.2e-4，后窗继续变好；但0 dB BER仍可退化，说明连续损失收益没有自动转化为硬判决错误减少。Trajectory和普通续训的BCE曲线几乎一样，尚无专属在线能力证据。详见阶段结果中的“同一累计轨迹的连续 Data BCE 复核”。

之后做了两项单因素在线消融。第一，最近4帧Reward Pilot聚合门控在单session 8帧早期筛查中未优于当前帧门控，0 dB有害接受更多，停止扩大该方案。第二，最近4帧Adapt Pilot梯度均值在两个新session、每SNR16帧下没有改善BER：0 dB更差，5/10/15 dB无变化，因此停止扩大简单均值重放。

同时把累计参数半径0.05扩大到0.10，确认0 dB后期更新确实会受原信赖域压缩；但两session结果方向相反，5 dB仍退化，不能证明扩大半径改善BER。随后完成32步多帧trajectory外层训练与同预算普通监督续训对照。新session上trajectory的Frozen−Online均值收益为0/5/10/15 dB的+1.63e-4/+8.14e-5/0/0，0 dB两个session方向相反且CI很宽；普通续训0 dB点估计反而更好。两臂连续Data BCE均改善且量级相近，但BER晚−早趋势不支持增长，trajectory没有可归因的专属在线增量。结果见full_head_trajectory_train32_v1_20261106。

因此不继续扩大BCE trajectory配置。已实现离线外层平滑BER目标（固定温度0.5），其单元测试验证梯度聚焦在靠近判决边界的错误符号；线上Adapt BCE和Reward门控不变。32步soft-BER trajectory臂及同配置、同预算普通监督臂均已完成训练，并在新session 76170/76171完成四SNR、16帧配对评估。结果归档见[soft-BER trajectory](results/full_head_softber_train32_v1_20261106/README.md)；普通监督checkpoint与逐帧评估见 `results/full_head_softber_supervised_train32_v1_20261106/`。soft-BER臂相对Frozen的0 dB平均收益为`+2.85e-4`，但两个session方向相反且描述性区间极宽；5/10/15 dB没有稳定BER收益。早晚窗变化在0 dB为负，其他SNR无变化。匹配普通监督臂0 dB也有正点估计，trajectory相对普通监督的收益差很小。因此软目标没有证明可重复在线BER收益，更没有证明收益随传输帧数增加；当前结论仍是全head Adam在线更新路线未通过开发证据门槛，不进入正式确认。

之后在新session 76200 上将冻结的soft-BER trajectory与匹配普通续训两臂扩至32帧、四SNR同波形配对，结果见[32帧延长筛查](results/full_head_softber_train32_v1_20261106/README.md)。0/5 dB两臂晚窗−早窗均为负，soft-BER没有优于普通续训；高SNR没有BER变化。因此不扩展当前全head Adam的12×128矩阵。下一步固定低维神经方向，验证跨帧状态不确定度/过程噪声更新是否改善长期增益；先实现数值与信息隔离测试，再以固定更新规则做短会话机制筛查，只有出现正向候选才扩大session和帧数。Data标签只用于离线外层目标及动作后事后评分，绝不进入在线候选或Reward门控。

低维状态滤波首轮已实现并完成四SNR smoke及同会话4帧GN配对，归档见[rank-1状态滤波筛查](results/rank1_state_filter_dev_v1_20261106/README.md)。状态均值仅在Reward接受时提交，拒绝时回滚；方差仍由合法Adapt Fisher更新。4帧中状态滤波与GN的Data BER相同，首个候选碰到0.05累计半径，后续多数被Reward拒绝。共同步长缩至0.005后，在新session 76205完成四SNR、8帧配对；两臂所有逐帧硬BER相同，接受数也完全相同，状态滤波没有可见增量或帧龄趋势。分块精确Jacobian曾解决显存但计算过慢，随后切换中心有限差分方向导数；线性解析对照与相关17项定向测试通过，测试GPU峰值低于2.7 GiB。固定未训练P0/Q配置没有理由继续扩大。下一步固定Pilot Fisher方向，基于已有可微rank-1 GN轨迹训练接口，离线学习跨SNR共享的状态先验方差、过程噪声和Pilot信息缩放；先验证外层梯度、Reward拒绝回滚、投影后参数/系数一致及Data/隐藏真值隔离，再决定独立session评估。在线Data标签仍只作状态更新后的事后评分。

## 2026-11-06 P0：soft-tail 训练/评估对齐

离线课程训练使用 `online_soft`（当前帧最后 116 个 Data 位置的模型软判决递推下一帧 tail）；评估函数 `next_pilot_tail` 只允许 Pilot 位置更新，而当前 `prefix` 布局帧尾全是 Data，因此该函数是空操作，tail 永远停在 acquisition 尾。主在线路径 `online_adaptation.py` 的 `pilot_visible_mask[-tail_len:]` 同样为空。

新增 `tail_update_mode`：`pilot_only` 保留历史语义，`dd_soft` 对齐离线训练。`dd_soft` 只读取接收波形与模型自身 soft 判决，不读取 Data 发送符号或 Data 比特；Frozen 与 Online 共用同一规则。结果见[tail 对齐归档](results/tail_alignment_v1_20261106/README.md)：

- 10/15 dB 的 Adapt Pilot BCE 从 `0.0282/0.0357` 降到 `0.0067/0.0003`；旧路径的 Pilot 损失主要来自错误的帧边界 tail。
- 5 dB 的 Adapt/Data 梯度余弦从 `0.456` 升到 `0.501`；0 dB 略降（`0.749→0.721`）。
- Frozen Data BER 在 0/5 dB 略好（`23.50%→22.69%`、`3.80%→3.70%`），10 dB 相同，15 dB 在极少数错误下略高。
- 10/15 dB 的 Adapt/Data 余弦仍接近零或负值；tail 对齐不能替代数据辅助信号。

统计功效分析见[功效与错误构成](results/power_error_composition_20261106/README.md)：历史 `1e-5` 量级点估计通常只是 1 个净错误，3 session × 16/32 帧严重欠功效。建议至少 10 个独立 session × 128 帧；0 dB 30–35、5 dB 10–15、10 dB 20–25、15 dB 10–15 个 session，对应可预注册的最小 ΔBER 约 `1e-3/5e-4/2e-4/1e-4`。

文献机制调研见[方法调研](results/method_survey_20261106/README.md)：首选低秩贝叶斯状态空间适配器 + 无标签 Data 的软边缘化模型似然（ELBO），数据辅助 CIR/相位精化作为物理侧补充。

P0 全量测试在 `PYTHONUTF8=1` 下 `1001 passed`。默认 Windows GBK 环境下，1 项子进程编码测试会失败；它与本改动无关，`PYTHONUTF8=1` 可复现通过。

下一步 P1：固定 tail 协议，等离线饱和/oracle 上界子 agent 结果归档后，实现"数据辅助软似然 + 低维状态后验"的单因素机制筛查。离线饱和结果尚未返回，不能当作已完成证据。

## 2026-11-06 离线饱和诊断与 P1 方向

[离线饱和诊断](results/offline_saturation_20261106/README.md)已完成（新开发 seed 91001–91005）：

- 训练预算/宽度接近饱和：续训 32/64 步、`iterations` 4→6、`d_model` 64→96 均无稳定改善；最大点估计约 `-1.8e-4`（5 dB，64 步）。
- 同帧 oracle head（Data 标签，K=64）在 5/10 dB 显著（`-2.34e-3`/`-7.81e-4`），但会话内留出帧复核后基本消失（区间跨零）；持久静态 head 修正不能稳定迁移。
- 真实 CIR 条件相对估计 CIR：5 dB `0.05283→0.04199`（`-1.08e-2`），10 dB `0.00700→0.00465`（`-2.34e-3`），相对 BER 约 20%–33%；15 dB 近似 0；0 dB 点估计 `-4.60e-2` 但 seed 方差大。

因此 P1 先做合法 CIR/相位精化，而不是持久 head。[Data 软边缘化 ELBO 诊断](results/data_aided_elbo_diag_v1_20261106/README.md)（2 seed × 6 帧）显示：5/10/15 dB 的 `cos(ELBO, Data)` 约 `+0.15/+0.14/+0.24`，而 Adapt/Data 在 10/15 dB 约 `-0.05/-0.01`；但 ELBO 梯度范数在 15 dB 仅约 `9e-4`，一次固定半径步进没有稳定 BER 收益，0 dB 方向为负。ELBO 适合作为辅助项或 CIR 精化目标，不能单独替代 Pilot。

新增实现：

- `training/data_aided.py`：Data 位置软边缘化 ELBO；标签隔离与梯度可达性测试。
- `training/data_aided_cir.py`：只用接收波形与模型 soft 判决的加权 LS CIR 精化；不读 Data 标签；与 Pilot-only 对照。
- `scripts/diagnose_data_aided_elbo.py`、`scripts/diagnose_data_aided_cir.py`。
- 定向测试 `tests/test_data_aided.py`、`tests/test_data_aided_cir.py`。

四模式 CIR 诊断（Frozen / Pilot-only / Data-aided / Oracle）正在运行；结果归档后再决定是否进入低维神经状态修正。开发证据，不进入正式确认。

## 2026-11-06 数据辅助 CIR 四模式诊断（负/中性）

[四模式诊断](results/data_aided_cir_diag_v1_20261106/README.md)（3 新开发 seed × 16 帧，`dd_soft` tail）：

- `data_cir`（Data soft 判决 + Pilot 相位线去旋转 + 加权正规方程）相对 Frozen：0 dB `+0.1528`（错误传播发散）、5 dB `+0.00035`（中性）、10 dB `-0.00057`（3-seed 区间跨零）、15 dB `-2.7e-5`。
- `pilot_sparse`（Pilot + 相位补偿 + top-24 稀疏 tap）也不稳定：0/5 dB 明显退化，后段发散。
- `oracle_cir` 真实 CIR 在 0/5/10 dB 仍更好（`-0.0343/-0.00705/-0.00109`），与离线饱和诊断一致：**CIR 头room 存在，但一次性 Data LS 没吃到**。
- 判断：不进入 N-DACC 的简单 Kalman 版本（坏观测会被平滑，不会变好）；下一步先做同帧 ELBO head 适配，必要时再做离线学习的去偏 CIR 估计器。
- 1-seed smoke 曾显示 data_cir 在 0/5/10 dB 改善，但 3-seed 复测不复现；再次说明单 seed/短会话不能作为机制证据。

## 2026-11-06 简单 CIR 估计器与同帧 ELBO 的负结果

- `pilot_ridge`（全 tap + Pilot 相位补偿 + ridge 向 acquisition 先验）在 1 seed × 8 帧 smoke 中全 SNR 明显退化：0 dB `18.2%→36.1%`、5 dB `3.68%→9.41%`、10 dB `0.407%→2.21%`。prefix Pilot 只有 108 个有效行、CIR 有 117 tap，`ridge=1e-2` 不足以约束欠定系统。停止该参数化。
- 同帧 ELBO head 适配 64 步（1 seed × 2 帧）没有复现同帧 oracle 增益；1 seed 样本太小，但方向不支持"简单 ELBO 步进就能吃到 oracle"。
- 判断：不再扩简单 LS/ridge 或单步 ELBO。下一步做**离线学习式 CIR 去偏映射**：输入 acquisition CIR、当前帧 Data LS、Data 置信度/相位特征，输出真实 CIR 残差；先用线性/低秩 map 验证 NMSE，再决定是否上小 MLP 与跨帧状态。Data 标签与真实 CIR 只用于离线训练/评分。

## 2026-11-06 贝叶斯递归 CIR（初步正向，待 3×16 确认）

简单 LS/ridge 失败的原因是忽略先验协方差与符号可靠性。新增[training/bayesian_cir.py](../training/bayesian_cir.py)：

- acquisition 帧构造归一化 CIR 先验均值与协方差；
- 每帧先做 Adapt Pilot 后验（已知符号），再做 Data 后验（模型 soft 判决）；
- Data 行噪声 `R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}`，低 SNR 错误符号自动降权；
- 后验协方差跨帧持久并加过程噪声，协方差收缩对应"收益随帧增长后饱和"。

1 seed × 8 帧 smoke（seed 95001）：

| SNR | Frozen BER | Bayes BER | 增益 | Frozen NMSE | Bayes NMSE |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.212402 | 0.214844 | −0.002441 | 0.1379 | 0.0902 |
| 5 dB | 0.046549 | 0.042806 | **+0.003743** | 0.0488 | 0.0231 |
| 10 dB | 0.005371 | 0.004232 | **+0.001139** | 0.0178 | 0.0091 |
| 15 dB | 0.000326 | 0.000326 | 0 | 0.0077 | 0.0051 |

这是本轮首个在 5/10 dB 同时出现正 BER 点估计与大幅 NMSE 下降的机制；0 dB 基本持平、15 dB 中性。**但仍是 1 seed × 8 帧，不能当作证据。** 3 seed × 16 帧诊断正在运行；结果归档前不进入 P2，也不宣称在线增益成立。定向测试 `tests/test_bayesian_cir.py` 4 项通过。

## 2026-11-06 共享 Bayes CIR 下的 head 增量（负）

[共享 Bayes head 诊断](results/bayes_head_diag_v1_20261106/README.md)（1 seed × 8 帧 smoke；Frozen 与 Online 共用 Bayes CIR、tail 与波形）：

- Online 每帧用 Adapt Pilot BCE 更新 head（4 步、半径 0.05、Reward BCE 门控），Data BER 相对共享 Bayes Frozen 在 0/5 dB 只增加 `+0.000163`（1 个净比特），10/15 dB 为 0；接受帧 7/5/3/7。
- 说明 Bayes CIR 已捕获 5/10 dB 的 CIR 相关 BER 头room；剩余的 per-frame head 修正没有稳定迁移到 Data。
- **归因结论**：当前最强在线增益是物理状态（CIR）增益，不是神经参数微调增益。AGENTS 契约明确纯 CIR 跟踪不能单独完成核心目标，因此不能把 Bayes CIR 单独写成研究点完成。
- 下一步：在共享 Bayes CIR 之上做学习式可靠性/噪声模型或后验不确定度条件化 FiLM；若仍无增量，需如实报告"物理在线增益成立、神经增量未成立"，并调整核心主张或结构。5 seed × 32 帧 Bayes CIR 增长-饱和筛查正在运行。

## 复现命令

```powershell
# soft-BER trajectory 外层训练（已完成；命令示例）
python pretrain.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --stage pilot_transfer --steps 32 --batch-size 1 --transfer-objective trajectory --trajectory-outer-loss soft_ber --trajectory-soft-ber-temperature 0.5 --resume <离线checkpoint> --save-dir <输出目录> --device cuda

# 匹配的普通监督续训控制（已完成；命令示例）
python pretrain.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --stage pilot_transfer --steps 32 --batch-size 1 --transfer-objective baseline --resume <同一离线checkpoint> --save-dir <输出目录> --device cuda

# 每个续训checkpoint的相同 Frozen/Online 评估
python scripts/prototype_head_rank1_adapt.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --pretrained <续训checkpoint> --basis-seed 78001 --seed-start <未见session起始seed> --num-seeds 2 --frames 16 --device cuda --branches frozen,full_head_adam --output <结果JSON>
```

此前全仓测试为992 passed；soft-BER改动的定向测试为131 passed，新增边界梯度测试单独复核为1 passed。soft-BER阶段结果已归档；新文档和结果文件尚待本轮检查、提交并推送。

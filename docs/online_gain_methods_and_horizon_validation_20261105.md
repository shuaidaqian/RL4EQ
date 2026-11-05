# 在线微调增益路线与随帧增长验证

## 起点已明确

离线神经均衡器已在既有 Level B 主矩阵中显著优于传统接收机，训练过程和 BER 见[离线训练与基线结果](offline_equalizer_training_and_baseline_20261105.md)。当前研究的缺口是：如何在这个强离线网络上实际更新少量神经参数，使 Online 超过同源 Frozen，并让配对收益在会话后段扩大。

近期 2 维 GN 开发路线没能解决该问题。v27 的 0/5 dB 正向点估计未在新会话复现；5 dB 的 Pilot/Data 梯度近乎不对齐。旧 v6 已试过把最近8帧 Pilot BCE 作平均，也没有产生可复现增益。因此不继续搜索 GN 步长，也不把简单增加 Pilot 历史帧数当作新算法。

## 算法调研与排序

|方向|为何匹配当前问题|本项目风险/边界|优先级|
|---|---|---|---:|
|持久的低维 Bayesian/RLS 神经参数跟踪|把 Adapt Pilot 的信息跨帧累积到参数后验均值与协方差；过程噪声/遗忘允许跟随慢时变信道。它与每帧重置曲率近似的点估计 GN 不同，直接对应“帧数增加后收益逐步积累”。|在线只用当前及过去已收到的 Adapt Pilot 波形/标签；参数维数由 Pilot 信息几何决定。要防后验过度自信和漂移遗忘不足。|1|
|Pilot-only 异步漂移检测与模块恢复|从留出验证 Pilot 的预测残差变化检测状态变点，只在证据提示漂移后更新下一帧，针对慢变化/突变浪费固定频率更新的问题。|验证 Pilot 不进当前拟合；检测只影响之后动作。检测阈值用训练/选择角色冻结。|2|
|过去帧 Decision-Directed 软伪标签|接收 Data 波形已经合法可见，使用模型自己的历史软输出作为伪标签，可在 Pilot 之外持续积累观测，直接检验“越传越多是否越能适配”。|只可用过去帧伪标签，不读发送 Data bits；错误会自我强化。置信规则必须离线冻结，当前验证 Pilot 仍独立守门。|3（高风险）|
|稀疏 trellis/BCJR 神经检测|结构化利用 ISI 和局部记忆，避免无约束全块更新。|116符号记忆下直接 trellis 状态空间不可行；先做截断/稀疏复杂度与 Pilot 可辨识性审计。|备选|

过去帧 Decision-Directed 的工程入口已做只读审查：[`deployable_frame`](evaluation/adaptability_audit.py) 会在控制前隐藏 Data bits、发送符号和仿真真值但保留 RX 波形；[`windowed_discrete_ppo.py`](training/windowed_discrete_ppo.py) 当前计算过整帧模型软输出和跨帧 soft-tail，但尚未缓存过去 Data 区域伪标签。若实现 DD，应新增只包含旧帧 RX、旧接收条件和 detached 模型软输出的有界缓存，严禁携带 Data 真值；至少添加 Data bits/tx/CIR/tail 毒化不改变 DD 控制的两帧测试，并检查缓存帧龄和 soft-tail 错误传播。

相关研究为机制依据，不是本项目效果证据：Park 等少 Pilot 离线/在线元学习（IEEE TSP, 2021, [DOI](https://doi.org/10.1109/TSP.2020.3043879)）；Cohen 等 Bayesian active meta-learning（IEEE TSP, 2022, [DOI](https://doi.org/10.1109/TSP.2022.3220035)）；Raviv 等 hybrid model-based online meta-learning（IEEE TWC, 2023, [DOI](https://doi.org/10.1109/TWC.2023.3241841)）；Uzlaner 等模块化异步漂移检测（IEEE TWC, 2025, [DOI](https://doi.org/10.1109/TWC.2025.3533959)）；Gusakov 等 streaming Bayesian deep receivers（IEEE TSP, 2026, [DOI](https://doi.org/10.1109/TSP.2026.3697255)）。这些论文的信道记忆、调制、Pilot结构和状态过程与本项目不完全相同，需要在当前固定配置里重验。书目和全文来源索引在原项目 `reference/core_online_adaptation_20261103.md`；该 `reference/` 目录未纳入本次代码提交。

本轮通过 Crossref DOI 元数据与本地 arXiv 作者公开全文核验书目信息；当前会话没有配置 nature-academic-search 的 PubMed/Crossref/arXiv MCP 搜索端点，因此没有声称完成 MCP 多库穷尽检索。

## 随帧增长的可检验定义

每个独立信道会话内，Frozen 与 Online 必须使用同一离线 checkpoint、同一波形、同一物理状态跟踪、同一接收尾递推、同一 Pilot/窗口和同步补偿。Online 的 Adapt 更新只读合法 Pilot；Reward/验证 Pilot 只做留出验收；Data 真值只在当帧动作、门控和回滚全部完成后用于离线评分。

把连续32帧作为一个窗口，定义会话`s`、窗口`b`的配对增益：

```text
g[s,b] = BER_frozen[s,b] - BER_online[s,b]
```

正值表示 Online 更好。每个会话至少128帧形成4个窗口；若机制通过开发筛查，长期试验扩展到1024/2048帧以覆盖当前配置的慢时变过程。另报告前缀累计增益`G[s,T]`（建议`T=1,2,4,8,16,32,64,128,...`）及每窗口 BER、参数距离、接受/拒绝/重置、事后有害更新数和计算延迟。

“收益随帧数总体增长”通过每个独立会话的窗口增益时间斜率、末窗增益和末窗减首窗增益检验。预期群体斜率为正，末窗 Online 增益为正，且末窗相对首窗改善为正；不要求噪声下每一帧或相邻每个窗口严格单调。先按会话计算，再用会话级配对/cluster bootstrap 得区间；帧数和 bit 数不是独立样本。四个 SNR 均报告，任何 SNR 的退化不得被平均值遮住。正式测试前冻结效应界限、置信区间、多重比较、样本量和停止规则。

## 下一组变量隔离实验

先用已经成立的 Pilot=256 离线 checkpoint（SHA256 见离线基线快照）做在线方法开发，不修改信道、PDP、同步分布和 Pilot 资源。当前 Codex worktree 缺少该权重；实验前应从原项目工作区恢复并核验指纹，不能悄悄改用 v27 的低维来源 checkpoint。

首个方法对照固定模型、会话、门控和预算，只改变更新器：

1. Frozen：神经权重冻结，但照常执行公平物理状态跟踪。
2. 当前 Pilot-only GN：已有基线，作为旧在线机制对照，不为其重新调参。
3. 持久 Bayesian/RLS Pilot 更新：Adapt Pilot 的逐符号残差和 Jacobian 递推更新参数后验；跨帧保留均值/协方差，固定遗忘/过程噪声由训练、选择角色确定，留出验证 Pilot 只守门。

先以每个 SNR 12 个开发会话、每会话128帧作机制筛查，全部四点报告；这只是开发样本，不是正式确认。只有当后验分支相对 Frozen 有正增益、末窗优于首窗，并且相对单帧 GN 显示正交互时，才用全新会话延长到1024/2048帧检查趋势。若 Pilot-only Bayesian 仍不能增长，再单独试过去帧伪标签分支；不把伪标签与 Bayesian 更新一次性混成一个方法。阈值/超参数只能在训练/选择角色冻结，评估 Data 不能反向选算法。

## 强离线 checkpoint 上的在线复核

2026-11-05已将现有 Pilot residual-RLS 接到有验证指纹的 `eme_bce_all_32_20260905_pilot256` 强离线模型上。8帧机制烟测发现原双窗 Reward 门控未提交任何更新；只改为单个 Reward BCE 窗口、继续保留硬错误不增护栏后，独立3-session×32帧开发筛查在5 dB出现初步正向窗口趋势，但全程平均增益仍为负，0 dB有害更新。64帧新session复核没有重现5 dB增长：0/5 dB整体退化、10 dB仅接近量化噪声的正点估计、15 dB BER不变，趋势区间没有支持收益增长。详细逐帧结果分别见[32帧开发筛查](results/strong_offline_rls_onewindow_32f_dev_20261105/README.md)与[64帧开发复核](results/strong_offline_rls_onewindow_64f_dev_20261105/README.md)。

因此暂停继续扩增该启发式 RLS/单窗变体，不以关闭硬错误守门或降低阈值制造更新率。下一路线回到离线内外循环：离线阶段训练表示/初始化，使合法 Adapt Pilot 更新后可泛化到未来 Data；在线只执行已匹配的低维确定性更新和留出 Reward 验收。第一步需明确当前 `pilot_transfer` 训练器能否复现 residual-RLS 的同一内循环；若不能，新增匹配的内循环，而不是直接拿现有 `phase_trend` 更新器冒充相同方法。之后再对强 checkpoint 做 Frozen、原启发式 RLS、离线适配初始化三臂同轨迹比较，并检验更长会话的分窗趋势。

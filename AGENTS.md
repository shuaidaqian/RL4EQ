# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

> **2026-11-06 起，当前研究规范以[研究路线 v4](docs/research_route_v4_20261106.md)为准。** v3 与全部旧结果保留原语义；v4 只调整当前目标、在线自由度、方法方向与验证判据，不改写历史实验结论。全仓文档索引与取代关系见[document_index_20261106.md](docs/document_index_20261106.md)。本文件下方的历史条目继续有效，但凡与 v4 冲突之处以 v4 为准。
>
> **2026-11-06 用户决策**：核心主张调整为"离线神经均衡器 + 合法在线贝叶斯物理状态（CIR）适配"。5×32 开发筛查显示 5/10 dB 物理增益 5/5 seed 同向且描述性区间不跨零；0 dB 32 帧内仍增长；15 dB 中性。原"在线神经参数微调必须优于 Frozen"降为独立、尚未成立的子主张；纯 CIR 跟踪可以单独作为当前核心的在线增益来源。信道、Pilot、SNR、数据角色与标签隔离不变。
>
> **2026-11-06 神经 PEFT 验证结论**：ELBO 低秩 Adapter 与不确定性 FiLM 两个机制均未产生 5/10 dB 可重复 BER 增益（0 dB 仅 1 比特，10/15 dB 为 0）；用户决定**接受负结果**。证据见[PEFT 探索计划](docs/peft_neural_exploration_20261106.md)与两个 smoke 归档。物理 Bayes CIR 结果不受影响。不再扩同一类机制；若重新开启，只有同帧 TTT、共享 Bayes CIR、离线重训/增广三个放松约束的选项，需重新授权。

## 当前目标与指令优先级

2026-11-06 用户确认 v4 的四项在线自由度（Data 位置无标签自监督/判决反馈、更多在线算力、更大参数集、跨帧会话级记忆/重放），保持“四 SNR 平均显著改善且每 SNR 点估计不退化”的现契约，离线饱和验证做中等深度，文档新事实源单独分支。核心方法方向改为 S-DANSA：共享物理状态、低维神经状态后验、Data 位置软边缘化模型似然、离线元训练四臂归因。v4 不改信道配置、Pilot 布局、SNR 或数据划分；旧实验不按 v4 追认成功。详见[研究路线 v4](docs/research_route_v4_20261106.md)。

2026-11-03 用户明确要求按新的分析调整研究路线。本文件及[研究路线 v3](docs/research_route_v3_20261103.md)取代此前“唯一 Level B + 必须 RL + 四 SNR 逐配置超越传统 + 后期优势扩大”的研究契约。旧配置、代码、实验结果保留原语义，不能按新标准追认旧实验成功。

核心目标：**离线学习可由少量合法在线观测适配的神经均衡器，在线真正更新少量神经参数，证明相比同源、具有相同物理状态跟踪的冻结网络有可归因、可重复的 Data 检测增益，并验证持续跟踪能力。**

2026-11-05 用户补充并纠正路线：现有离线神经均衡器已在既有 Level B 多 seed 主矩阵中明显优于传统 baseline；先整理训练过程与结果，研究重点是 Online 相对同源 Frozen 的额外神经参数收益。新增的持续适配假设是**随着同一会话传输帧增加，Online 相对 Frozen 的配对 BER 收益总体扩大**。不能把“逐帧不恶化”作为代替，也不能要求随机误码下每一个相邻帧严格单调；必须用预先分窗趋势、末窗与早窗差、会话级统计检验该增长假设。

2026-11-03 用户进一步明确：**不要改信道配置，按研究路线 R0–R6 推进。** 该指令覆盖此前短/中记忆机制场景的实施建议。当前 R1/R2 直接使用已有 116 符号长回波配置；不得新增短记忆 profile、缩短记忆、改变 PDP/漂移/同步分布或调整 Pilot/接收窗口来获得收益。可以修改神经结构、离线训练和合法在线更新算法，并在相同信道与资源协议下比较。后续 R4 单因素外推与 R5 物理标定若涉及信道变化，须先明确对应范围，当前不自动生成变体。

- EME 长回波是重要应用背景与迁移验证对象；Level B 不再是所有机制设计与验收的唯一入口。
- RL 是可选调度扩展，不是核心方法或核心成功的必要条件。先验证确定性神经微调，再决定是否研究调度；不得继续把旧 PPO 大矩阵作为默认下一步。
- 物理状态跟踪的改善不等于神经参数微调的改善。必须分别测出物理状态适应、离线表示和在线梯度更新的贡献。

## 不可放松的信息与公平性约束

- 在线允许读取已收到的波形；禁止读取 Data 发送符号、Data 比特、仿真真实 CIR/相位/CFO 等隐藏真值来更新、选择、验收、回滚或设定阈值。
- Adapt Pilot 用于更新。现有 Reward Pilot 保持为独立动作后留出集，只用于验收、reward 与评估，不回流内循环拟合。无 RL 时称为“验证 Pilot”，历史字段可继续叫 Reward。
- Data 标签只用于离线训练、离线开发/校准、独立测试的事后评分；这些角色的数据、seed 和访问时序必须隔离。外层未来标签不得泄漏到在线更新。
- 真状态接收机只作隔离的离线诊断，不能给可部署分支提供真值。`Perfect-CSI Block`、`Fixed CG-BPSK-DD Block Detector` 不自动是最优性能界，不进入可部署主方法比较。
- 同波形比较必须匹配 Pilot 预算、接收窗口、acquisition、同步补偿与物理跟踪协议，报告在线参数数目、更新次数、计算量和延迟。Frozen 不必执行无用梯度来消耗相同时间，实际计算差异必须报告。
- 传统 baseline 不使用神经网络或 RL，只用合法观测与传统规则，提供合理同步/自适应补偿并独立校准。允许多个神经对照和确定性 PEFT 方法，不再限定只有 Proposed 能用神经网络。
- Frozen 表示网络权重冻结，仍可依协议逐帧进行 Pilot 条件适应；不得人为削弱 Frozen 制造在线收益。

## 场景与建模规则

- 当前机制与结构验证直接在已有 116 符号长回波配置内进行，不以短/中记忆场景作为前置条件。后续迁移和物理应用分别报告，不同 profile/Level 不混合成一个主平均。
- 当前仅在已有配置下使用同分布信道实现；CIR 漂移、CFO、慢相位、强径数、弥散比例的单因素外推属于后续 R4 范围，不能据此覆盖用户“不改信道配置”的要求。
- 首轮保留前缀 Pilot、单载波线性 BPSK、整帧缓冲的非因果块均衡。现有长回波的 116 样本/11.6 ms 后缀公平使用。Pilot 布局与采样率等不是物理定律，后续变更需要单独版本和同资源消融，不能静默修改历史协议。
- 开发与确认均覆盖0/5/10/15 dB四点；用户确认全部作为主要运行点，不根据开发结果删点。核心假设是在线微调**必须优于**同源、同物理状态跟踪的离线冻结网络；证明不退化只是必要护栏，不算研究成功。在线允许BER退化量为0，配对差定义为`BER_online - BER_frozen`。主要优越性效应、逐SNR零退化置信界/多重比较、目标吞吐及物理频段/带宽仍需正式确认前冻结。
- 11.6 ms 时延支撑不等于当前全部 PDP、强径数、弥散比例和 1200 s 相关时间已被测量支持。每个物理参数标记文献/测量来源、建模假设或机制变量。
- 不为获得神经胜出而调整信道。现有 EME profile 的严格尺寸与物理字段校验继续生效；不新增短记忆机制 profile，也不修改已有信道生成行为。
- 不恢复旧 A2C、逐符号 PPO、Data Oracle、多载波、MIMO 或 RIS 路线。非线性、编码和高阶调制当前不扩展。

## 方法与离线训练

- 优先物理模型辅助的神经块均衡；同步状态、物理信道状态与可训练神经参数分开定义。
- 在线自由度由合法 Pilot 信息矩阵的秩、条件数、变化结构与留出泛化决定，不能用“占整网比例小”代替可辨识性。
- 低维神经子空间、Adapter/FiLM/LoRA/context 可作为**独立神经子主张**的候选；2026-11-06 决策后，当前核心允许合法在线贝叶斯 CIR 物理状态适配单独作为在线增益来源；神经参数在线更新另行验证。
- 离线内循环复现在线的损失、历史、优化器、正则、投影、持久保留和更新时序；外循环评估当前与后续 Data 泛化。可学习表示、子空间和初始化，不局限于旧 34 个尺度参数。
- 用匹配数据与预算的普通监督/更新后训练四臂比较，排除仅继续训练或改变离线模型带来的收益。
- 首先使用固定步长、近端或带遗忘等确定性更新。累计距离约束、逐步信赖域和过程正则需要单独验证，不能无证据取消保护。
- 若以后引入 RL：只调度离散安全动作、作用短窗口，使用窗口级验证 Pilot 反馈；不逐 bit 判决，不直接生成完整高维参数增量，并与固定规则和简单漂移调度比较。

## 证据与验收

- 分开验收：核心神经微调增益、传统方法竞争力、EME 迁移/物理应用、可选调度增量。后一项失败不自动否定前一项；前一项未成立不能靠后者绕过。
- 核心通过必须由独立信道级配对结果证明物理 Bayes CIR 适配优于同源 acquisition-CIR Frozen；神经参数子主张需共享 Bayes CIR 后另行验证。仅“不劣”或“未发现差异”不算成功。四 SNR、逐 seed 和完整会话均报告，并满足每点最大 BER 退化量为 0 的预注册安全护栏。
- 持续跟踪将“Online−Frozen 收益随帧数总体增长”作为明确待验证假设：按固定窗口报告配对 BER 增益、拟合会话内时间斜率并比较末窗与早窗；同时报告稳态 BER、漂移后恢复、失稳/有害更新与成本。随机信道误码允许局部窗口回升，但不能用全程平均掩盖增长趋势失败。
- `BER_data < 0.01` 仅是可能的系统指标；应用要求未确定时不能冒充已冻结的正式目标。
- 正式测试前冻结主要区间、效应量/退化界限、统计方法、多重比较处理、样本量、会话与漂移定义、停止规则、配置与 checkpoint。开发集不得反复当作独立确认集。
- 旧门槛未通过的事实保留。新路线调整本身不代表新核心目标已经达成。
- 代码入口仍是 `pretrain.py`、`online_train.py`、`compare.py`、`calibrate_channel.py` 和 `pytest`。旧方法分组不是新路线已实现的证明；按[首轮审计计划](docs/adaptability_audit_plan_20261103.md)推进，已完成长回波开发审计；下一步在同一信道内选择可辨识结构并实现匹配离线训练。

## 验证与阶段交付

- 代码改动运行适当测试；纯路线/文档调整检查内容、JSON 和链接，不为此重复完整 GPU 训练。
- 默认测试：`python -m pytest -q -p no:cacheprovider`。本工作树尚无 `.venv-gpu`；可用 GPU Python 位于 `D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe`，运行前检查环境，不把机器路径当项目依赖。
- 跨帧 `pretrain.py` 当前使用 `--stage pilot_transfer`；共同历史配置的旧确定性 Pilot 兼容入口不支持该目标，不能仅用 `--method pilot` 冒称新确定性方法已实现。旧 PPO 复现需显式 `--method ppo`。
- 新确定性开发审计使用 `online_train.py --method audit --audit-manifest configs/audits/adaptability_long_anchor_v1.json`，条件由清单固定，禁止混用旧切片参数。长回波实现见[阶段说明](docs/deterministic_audit_implementation_20261103.md)；它尚不代表新结构、匹配离线训练或正式确认已完成。
- 固定信道检测子空间新增 `online_train.py --method subspace_audit` 与 `pretrain.py --stage detection_subspace`，使用独立清单 `configs/audits/detection_subspace_long_v1.json`。检查通过后才训练，预算与数据角色见[实现说明](docs/detection_subspace_implementation_20261103.md)；来源网络冻结，在线仅更新2/4/8维神经检测系数，实际结果另行归档。
- R2 第二轮允许使用 `configs/audits/detection_subspace_long_v2_reset_v1.json` 的验证Pilot初始系数回退规则；它只比较当前、候选和离线初始神经系数，不读取Data标签，`candidate_accepted`与`reset_to_origin`必须分开报告。信道和观测资源不变，未通过开发证据前不进入R3。
- R2 第二轮结果已归档于 `docs/results/detection_subspace_long_v2_reset_v1_20261103/`：回退实际触发但没有可归因、可重复的Data增益；验证Pilot BCE与Data BER在接受候选中的一致性很低。下一步先做验证统计量/外层目标校准的只读诊断，再决定结构或训练单因素，禁止按此结果进入R3或扩大同结构seed矩阵。
- 只读连续Data BCE诊断修正了BER并列导致的判断：验证BCE与候选Data BCE方向在0 dB约86%、5 dB约65%一致；R2第三轮清单 `configs/audits/detection_subspace_long_v3_decay_v1.json` 固定0.5离线锚点收缩，收缩后再守门，内外循环一致，沿用开发seed仅作机制对比。不得将其视作独立泛化证据；运行结果需归档后再决定下一项R2单因素。
- 第三轮结果已归档于 `docs/results/detection_subspace_long_v3_decay_v1_20261103/`：部分有害保留减少，但核心可归因在线Data增益仍未成立。R2继续，不能进入R3或正式确认。
- 第三轮只读梯度诊断：B臂5 dB的Adapt/Data BCE梯度余弦均值约0.055，15 dB Adapt梯度范数约2.6e-6且方向中位余弦近零。下一R2迭代优先验证物理模型辅助Pilot目标的尺度/归一化与离线内外循环匹配；Data梯度只用于事后诊断，不进入可部署控制。
- 第四轮固定清单 `configs/audits/detection_subspace_long_v4_tempered_adapt_v1.json` 仅把Adapt BCE logits除以0.5，验证仍使用原始BCE/硬BER；不按SNR调度。全部SNR必须报告，seed复用只作开发机制对照。实现后定向65项通过，GPU审计与四臂结果待归档。
- 第四轮结果已归档于 `docs/results/detection_subspace_long_v4_tempered_adapt_v1_20261103/`：固定温度没有改善B臂Data BER，0/5 dB均值退化，交互为负，温度只保留为失败开发消融；R2继续，不能进入R3。
- 第五轮固定清单 `configs/audits/detection_subspace_long_v5_temporal_basis_v1.json` 只将逐符号MLP基改为局部卷积基（temporal_conv_v1），使用v3更新规则，不带温度缩放；信道、Pilot、seed、A/B预算不变。定向67项通过，GPU几何和四臂评估待运行。
- 第五轮结果已归档于 `docs/results/detection_subspace_long_v5_temporal_basis_v1_20261103/`：8维通过审计，B在0/5 dB有同向正点估计但区间跨零；10/15 dB没有可确认增益。该结构只达到“值得独立确认前继续冻结并核对”的开发证据，不等于R2/R3完成。R3启动前必须冻结应用主要SNR、退化界限、统计方法和样本量。
- 第五轮后完整pytest为908 passed；当前路线文档与结果归档同步到该状态。
- 第六轮清单 `configs/audits/detection_subspace_long_v6_adapt_history8_v1.json` 只将内循环Adapt BCE扩展到最近8帧合法Pilot的均值；当前帧验证Pilot仍独立守门，外层与在线历史长度/时序一致，其他信道、Pilot、结构、预算和开发seed不变。定向68项通过，GPU几何审计、匹配四臂与768条开发评估已完成并归档于 `docs/results/detection_subspace_long_v6_adapt_history8_v1_20261103/`。B臂0/5 dB增益点估计约 `+1.36e-5`但3-seed区间跨零，10 dB约 `-1.36e-5`且区间跨零，15 dB无足够错误判断；B-A交互未建立稳定正证据。子空间更新约40 ms不含端到端接收和来源跟踪。R2继续，不进入R3。
- 第六轮只读守门诊断 `scripts/diagnose_subspace_gate_alignment.py` 与结果 `docs/results/detection_subspace_long_v6_gate_alignment_v1_20261103/` 只比较最终保留Pilot BCE与动作后Data BCE/BER；每SNR三个重复开发seed，仅作机制诊断。会话内Spearman均值在0/5/10/15 dB约 `+0.11/-0.14/-0.34/-0.60`，5/10 dB接受候选后的均值Data BCE变化为退化方向。守门校准是R2待修瓶颈之一；原轨迹无梯度向量，Adapt/Data梯度方向需单独只读诊断。Data标签继续禁止在线控制。

- 第十六轮12维开发结果见 `docs/results/detection_subspace_long_v16_dim12_frozen_error_v1_20261105/`：12维通过预定Pilot信息条件数门槛，但Frozen错判条件化训练在0/5/10/15 dB的B臂在线BER均与Frozen相同。只读Data边界诊断显示候选会改善约37%–57%的错误边距，却也伤害其余错误；3步累计位移上界内理论可达比例仅约1.2%/4.0%/11.5%/12.2%，且候选仅翻转1个错误。逐帧Data BCE oracle在总半径0.8内四SNR均有小幅聚合BER改善，说明该表示有少量修错空间；下一步聚焦合法Pilot梯度/预条件与跨符号更新冲突。oracle脚本 `scripts/diagnose_subspace_data_oracle.py` 仅作离线诊断，绝不进入部署或在线门控。全量测试 `942 passed`，oracle专属定向测试41项通过。
- 第十七轮全局梯度传输矩阵开发结果见 `docs/results/detection_subspace_long_v17_dim12_gradient_transport_v1_20261105/`：在12维v16表示与更新上只增加单位阵初始化的有界对称正定梯度传输矩阵；四SNR B臂在线BER仍与Frozen相同，传输矩阵偏离单位阵Frobenius范数约0.0226、条件数约1.023，未形成有用全局预条件。定向测试42项通过，全量测试943项通过。先前提出的Pilot局部影响审计已完成，见下条。
- v17之后的只读Pilot→Data Jacobian距离诊断已归档于同目录：Data与Adapt Pilot最近距离约33符号，33–48、49–64、65+区间的平均绝对基行余弦约0.79–0.81，没有随距离显著衰减。该值只说明共享系数输出作用高度相关，不等于梯度方向或部署效果。下一结构候选改为Pilot推断上下文、接收波形特征条件化Data位置修正；先审计新增输出的合法Pilot信息秩与可辨识性，再训练。诊断脚本为 `scripts/diagnose_pilot_data_jacobian_locality.py`。
- v18直接堆叠127符号感受野空洞卷积的信息审计已归档于 `docs/results/detection_subspace_long_v18_dilated127_audit_v1_20261105/`：仅2维满足条件数门槛，12维最坏条件数约14450，因此未训练。v19在空洞卷积加入0.1残差旁路后，4维最高合格，12维最坏条件数约633；4维开发四臂B在线增益在0/5/10/15 dB为 `+1.36e-5/-1.36e-5/0/0`，区间跨零，未证明在线优势。见 `docs/results/detection_subspace_long_v19_dilated127_residual_v1_20261105/`。v18/v19架构改动后全量测试待运行。
- v20 Pilot上下文条件化逐位置Adapter见 `docs/results/detection_subspace_long_v20_pilot_context_adapter_v1_20261105/`：零上下文精确保留Frozen输出，在线2维上下文只由Adapt Pilot更新，Data位置修正依赖已接收特征。Pilot信息审计最高合格2维。B臂在线增益0/5/10/15 dB为 `-1.36e-5/+4.07e-5/-1.36e-5/0`，A臂平均点估计相同，B−A平均交互为0；未证明在线优势。上下文动态logit接口已扩展到内循环，边界诊断需调用非线性`model.logits`，不能用原线性`basis @ coefficients`近似硬输出。定向49项、全量950项通过；下一步只读分解正负SNR的Adapt梯度与Data oracle对齐，再选择更新机制。
- v21只把v20 Adapt历史长度从8改为1，结果见 `docs/results/detection_subspace_long_v21_pilot_context_history1_v1_20261105/`。B臂0/5/10/15 dB在线增益为 `+1.36e-5/+2.71e-5/0/0`，A臂逐点完全相同，交互均为0；不能归因于B臂离线元训练。v21梯度诊断显示当前帧Adapt/Data余弦在5/10 dB约`+0.150/+0.091`，但实际增量在这些点仍未沿Data下降方向。下一步固定v21 checkpoint选择机制，用新开发seed作一次跨seed复测；仍不属于正式确认。v21当前全量测试待运行。
- v22固定v21机制，在新开发seed 121400/121401/121402复测，见 `docs/results/detection_subspace_long_v22_pilot_context_history1_crossseed_v1_20261105/`：0 dB B臂退化约`-4.07e-5`，5/10/15 dB无增益；A臂0 dB也退化，B−A没有稳定正交互。v21原开发seed上的小正点估计未复现；停止把单帧历史作为有效机制。下一步审计2维上下文的阻尼Gauss–Newton/近端Pilot更新，不改变信道/资源或标签边界。v21/v22后的全量测试待运行。
- 系数梯度只读重建 `scripts/diagnose_subspace_gradient_alignment.py` 与结果 `docs/results/detection_subspace_long_v6_gradient_alignment_v1_20261103/` 使用相同轨迹波形指纹、缓存和B checkpoint逐帧重建更新前系数。Adapt/Data余弦在0/5/10/15 dB约 `0.408/0.236/0.224/0.169`；15 dB Adapt梯度约 `3.51e-6`，Data梯度约 `1.54e-3`；验证/Data余弦在5/10 dB约 `0.056/-0.012`。这些是三个重复开发会话的机制数据，不是确认。下一步只读筛查统一Adapt logit温度/尺度，验证守门不变；若无全SNR可行尺度则换更新表示/目标，禁止按SNR调度和改信道。
- 统一Adapt温度只读筛查已完成，结果见 `docs/results/detection_subspace_long_v6_temperature_sweep_v1_20261103/`；温度2在四SNR的Adapt/Data梯度方向均优于温度1，按四点最差对齐选择温度2进入一次匹配开发。v7清单 `configs/audits/detection_subspace_long_v7_adapt_temperature2_v1.json` 与第七轮归档 `docs/results/detection_subspace_long_v7_adapt_temperature2_v1_20261104/` 已完成几何、A/B训练和768条配对开发评估。四SNR配对区间均跨零，B在10/15 dB仅各接受一次；梯度对齐改善没有转化为可靠BER增益。核心目标未完成，不能进入R3。下一步只读定位验证Pilot的候选风险统计与跨帧有效更新频率，先分析已有开发轨迹；不得用Data校准在线阈值或按结果增加同类seed。
- v7门控、暂存候选和Data可达性只读分析已归档于 `docs/results/detection_subspace_v7_gate_diagnostics_20261104/`。区块置信下界门控显著压低更新率但未一致减少Data有害候选；宽松门控在10/15 dB接受更多更新却仍有较多Data有害候选；8帧暂存也无稳定增益。隔离Data副本优化显示8维检测子空间有局部可达改进方向，但不代表Pilot可找到。停止继续调门槛，下一步检验离线外循环的硬接受分支梯度阻断：线上前向保持完全相同的验证Pilot硬守门，训练反向对拒绝动作使用显式直通梯度代理，单独做四臂开发验证；Data不参与在线控制，信道和资源不变。
- v8直通梯度实验已归档于 `docs/results/detection_subspace_long_v8_surrogate_gate_v1_20261104/`：未建立在线BER增量。v9清单 `configs/audits/detection_subspace_long_v9_gradient_transport_v1.json` 仅新增单位阵初始化的8维对称正定梯度传输映射；在线仍只更新8个神经系数并使用原投影/验证硬门控。矩阵初始与v8完全等价，离线A/B各64步，开发集覆盖四SNR和768条配对记录。归档见 `docs/results/detection_subspace_long_v9_gradient_transport_v1_20261104/`：B映射偏离单位阵范数约0.057、条件数约1.056，但B臂四SNR均无Data BER变化；A臂0/5 dB点估计一正一负且区间跨零，10/15 dB无可判断变化。v9核心目标未完成，不能进入R3。
- v10上下文非线性更新实验已归档于 `docs/results/detection_subspace_long_v10_context_update_v1_20261104/`。零输出初始化的17-16-8 MLP根据最近8帧合法Adapt梯度方向、历史分歧和幅度输出有界增量，Data不进入在线接口。A/B参数与预算匹配，四SNR共768条开发记录；B接受次数增加至0/5/10/15 dB的23/24/4/1，但全SNR的Data BER均无变化，Data BCE点估计方向也不一致，选择损失略差于A。梯度对齐和门控诊断仍显示5/10 dB验证/Data方向弱。全量pytest为926 passed，核心目标未完成，不能进入R3。下一步固定线上Adapt更新器和验证守门，只做单因素离线外层目标比较：标准Data BCE与预先规定温度的平滑误码/边界损失；保持四SNR、数据角色和预算不变，不能按SNR调参或用Data校准在线门控。
- v11平滑误码外层目标已归档于 `docs/results/detection_subspace_long_v11_smooth_ber_outer_v1_20261104/`：线上规则保持v10，仅将离线外层与checkpoint选择目标改为统一T=1.0的sigmoid平滑BER。Adapt/Data梯度余弦升至约0.629/0.497/0.339/0.233；0dB每个开发会话约减少1个误码，5dB点估计退化且跨seed异质，10/15dB无变化；B选择损失仍略差于A。0dB极窄区间由三个开发会话相同的离散单bit变化造成，不是稳定性或独立确认。全量pytest为928 passed。核心仍未完成，不能进入R3。下一步不再改训练损失或按开发Data调门控；先开发一个仅根据动作前后合法验证Pilot统计量评估更新风险的离线训练门控模型，并以隔离选择/评估会话检查误拒/误收及Data事后风险，在线只允许模型读取Pilot波形和Pilot标签。
- v12 Pilot-only风险门控已归档于 `docs/results/pilot_validation_risk_gate_v1_20261104/`。线性岭回归用131个训练候选拟合动作Data BCE改善，66个选择候选上Pearson约0.465，固定0阈值产生6个误收和8个误拒。评估中0/5/10/15 dB分别拒绝4/7/1/0个原硬门控动作；0dB有害接受数不变且误拒4个有益动作，5dB有害动作只少1个且误拒6个，10dB剩余接受动作仍全部有害，四SNR BER均未改变。不能设为默认在线门控，核心目标未完成。下一步只读分析Pilot风险分数对原硬门控拒绝候选的离线排序，评估是否值得做独立的Pilot-only替代门控机制；不得用本轮评估Data调阈值，也不得改信道/资源。
- v13风险替代门控反事实已归档于 `docs/results/pilot_validation_risk_gate_replacement_v1_20261105/`，实现入口 `scripts/train_pilot_risk_gate.py`，固定清单 `configs/audits/pilot_validation_risk_gate_replacement_v1.json`。复用v12风险回归器和零阈值，比较原硬门控、风险过滤、风险替代；四SNR BER增益分别为硬门控 `+1.36e-5/-1.36e-5/0/0`、过滤 `+1.36e-5/-1.36e-5/0/0`、替代 `0/0/0/0`。替代策略没有带来BER增益，高SNR放行更多事后Data BCE有害动作；当前风险门控分支停止，不作为默认策略，不进入R3。v13定向31项、全量pytest 932项通过。核心成功必须证明在线优于离线冻结网络，BER退化界限0只是护栏；候选统计方案、样本量、会话和停止规则见 `docs/protocols/confirmation_criteria_draft_20261105.md`，尚待正式冻结。判据冻结前不使用独立确认集，也不按开发Data调阈值。
- v14第一项R2后续实验已归档于 `docs/results/detection_subspace_paired_online_gain_v1_20261105/`。唯一改动是离线外层显式联合优化Frozen/Online平滑BER并惩罚Online落后；沿用v11结构、更新器、门控、seed和预算。B臂相对自身Frozen的0/5/10/15 dB BER增益为 `+1.36e-5/-1.36e-5/+1.36e-5/0`，所有可估95%区间跨零；0和10 dB正点估计各来自一个开发会话约少1个错误，15 dB无更新。选择损失B略差于A，配对外层目标未产生可重复在线收益，核心目标仍未完成。只读Adapt损失梯度筛查见 `docs/results/detection_subspace_adapt_loss_alignment_v1_20261105/README.md`：当前T=2 BCE在选择会话整体最好，top-hard/focal没有全SNR改善，15 dB梯度范数约`7.56e-8`。Data边界只读诊断见v14归档：单步局部步幅内可达错误占0.56/1.86/3.60/4.88%，三步上界内占1.81/5.67/18.00/21.95%；实际Pilot候选仅翻转4/1/0/0个错误，同时对错误边距有正有负作用。下一项在固定信道与开发seed上验证离线Frozen错误条件化Online外层目标，让Adapt更新学习修复冻结网络错误且守住正确位边距；Data分桶仅用于离线训练/选择。正式确认不启动。
- v15 Frozen错判条件化Online外层目标已归档于 `docs/results/detection_subspace_frozen_error_outer_v1_20261105/`。唯一改动是离线外层依冻结Data错判分桶，平衡训练Online对错误/正确符号的平滑误码并保留Frozen锚点。四SNR A/B各自Online BER均与Frozen完全相同，B−A在线增益交互全为0；接受更新没有改变Data硬判决。v15全量pytest `934 passed`。此损失不保留为默认。下一项只审计16维子空间在原Adapt资源下的Jacobian秩、条件数和输出作用；合格后才决定是否训练。在线增益未成立前不做正式确认。
- 阶段成果整理代码、文档、配置与结果指纹，提交并推送。没有实验或测试证据时明确写“未运行/待实现”，不得用计划替代完成证据。
- v24可微阻尼Gauss–Newton内循环匹配训练、v25配对Online增益外层目标均未产生稳定在线收益；v26只将近端权重0.1降至0.01，步幅显著增大但BER仍无一致正增益。报告分别见 `docs/results/detection_subspace_long_v24_gauss_newton_meta_v1_20261105/`、`docs/results/detection_subspace_long_v25_gauss_newton_online_gain_v1_20261105/` 和 `docs/results/detection_subspace_long_v26_gauss_newton_proximal_audit_v1_20261105/`。下一步固定prox=0.01做匹配A/B离线训练；保持Pilot/信道/窗口与在线门控不变，使用新开发seed，不进入正式确认。
- v27已完成prox=0.01下匹配A/B离线训练，使用新开发seed `121700/121701/121702`，结果见 `docs/results/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1_20261105/`。B臂0/5 dB Online−Frozen增益点估计`+4.07e-5/+5.43e-5`、B−A交互`+6.78e-5/+4.07e-5`，但增益区间跨零；10/15 dB无变化。只读诊断发现B臂5 dB Adapt/Data梯度平均余弦约`−0.018`，GN候选平均增加Data BCE约`7.93e-5`；15 dB Adapt梯度弱且验证梯度近零。冻结v27 checkpoint在独立seed `121800/121801/121802`复核后，0 dB B增益仅`+1.36e-5`且B−A为负，5 dB B退化`−8.14e-5`且B−A为负，10 dB小幅退化，15 dB无更新；复核见 `docs/results/subspace_v27_checkpoint_replay_v1_20261105/`。因此停止扩展当前单帧GN/prox=0.01设置；下一步只读分析合法Adapt Pilot残差的跨会话可辨识性，再设计不同更新表示。核心目标未达成，不进入R3或正式确认。
- v27冻结checkpoint复核入口为 `scripts/replay_subspace_checkpoints.py`，固定配置 `configs/audits/subspace_v27_checkpoint_replay_v1.json`，使用新的开发seed而不重训或重选checkpoint；定向测试2项通过，全量pytest `959 passed`。复核仍是开发证据，非正式确认。
- 2026-11-05强离线checkpoint上的启发式Pilot residual-RLS复核：单窗Reward BCE门控可增加实际神经参数提交，但64帧新session未出现可重复的Online−Frozen增长，0/5 dB整体退化、10 dB小正值跨零、15 dB BER不变。完整结果见 `docs/results/strong_offline_rls_onewindow_32f_dev_20261105/` 与 `docs/results/strong_offline_rls_onewindow_64f_dev_20261105/`。暂停该启发式更新器扩展；下一步审计/实现与在线更新匹配的离线内外循环，严格沿用当前信道、Pilot与信息隔离协议。

- 强checkpoint上的32步单帧 `pilot_transfer` A/B也已筛查：A/B Frozen BER四点相同，0/5/10 dB在线收益完全相同，B−A交互为零；5 dB晚窗开发信号两分支相同，全程收益近零。见 `docs/results/strong_checkpoint_pilot_transfer_meta32_20261105/`。下一步实现与在线时序匹配的多帧持久PEFT内循环，外层评价多个后续Data窗口；不得将单帧更新目标冒充持续收益训练。

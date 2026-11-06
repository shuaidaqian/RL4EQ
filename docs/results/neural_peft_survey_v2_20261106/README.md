# 强物理基线之上的神经在线适配：机制、负结果与候选（survey v2）

> 日期：2026-11-06
> 范围：2023–2026 年（含 2026 预印本）公开文献，重点不是“再估计一次 CIR”，而是回答：
> **当 Bayes CIR 已经把 CIR 相关头 room 吃掉之后，为什么神经在线适配仍然没有可重复增量？**
> 以及在这种基线下，哪些作用点/信息源仍被文献证明有头 room。
>
> 本报告只做文献与机制分析，不改变仓库代码、信道、Pilot、窗口或资源协议。
> 所有“增益”均为原文报告值，不等价于我们的设置；核验等级逐条标注。

---

## 0. 结论先行

1. **这不是“损失函数没调好”，而是贝叶斯下界问题。** 在线适配只能缩小“相对 Bayes 最优预测器的超额风险”（Bayes gap）；若基线已经接近 Bayes 最优，剩余误差是**不可约后验方差**，任何参数更新都无法降低。Wakayama（ICML 2026）把 TTT 写成核岭回归中的隐式贝叶斯推断，并给出精确的 spectral Bayes-gap 分解：适配只有在“滤波器与测试时 SNR 匹配”且“更新方向与 query 相关特征方向对齐”时才降低误差；固定子空间不匹配会留下**不可约的误差地板**，且“拟合 prompt 好”不等于“query 变好”。[W26]
2. **物理估计侧也已经接近信息下界。** 对 Gaussian WSSUS 信道，严格因果 pilot-to-prediction 的 LMMSE 基准**等于 Bayesian Cramér–Rao 界（BCRB）**，误差可分解为“Doppler 决定的不可约时间预测误差”与“pilot 密度/SNR/相关时间决定的估计误差”；五种神经架构中 Transformer 与 SSM 在多数 Doppler 区间**紧贴该基准 5% 以内**，结论是进一步增益主要取决于 pilot 密度、SNR 或预测时域。[L26]
3. **线性高斯模型下，神经估计器最多逼近 MMSE，不能超过它。** ReLU DNN 是分段线性函数，可万能逼近 MMSE；在线性信号模型下只能接近 LMMSE，显著超过 LMMSE 的情形来自非线性、非高斯或模型失配。[H20]
4. **在完美 CSI / 匹配分布下，神经均衡器只能匹配 BCJR/LMMSE，不能超过 BER 最优检测器。** 小参数 NN 均衡器容易落入局部极小、**连 LMMSE 都超不过**；用 LMMSE 初始化并渐进训练后可以超过 LMMSE、缩小与 MAP/BCJR 的差距，但仍不能超过 BCJR。[R24]
5. **大多数“在线适配增益”本质上是“分布偏移恢复增益”。** Obeed & Jian 的 DMRS 持续学习在**匹配分布**下神经接收机已接近“不可实现的完美 CSI LMMSE”；突然偏移后固定模型急剧恶化，在线微调恢复到接近原始性能，连续漂移场景中在线模型保持 <2e-3 而固定模型恶化到 5e-2；[O26] “Training Coverage” 在标准 5G NR PUSCH 上更直接：**训练包络内两种学习接收机都只是匹配 genie LMMSE**，包络外（2000 ns RMS 时延扩展、单 DMRS、30 m/s）才有 0.90–4.08 dB 优势。[T26]
6. **我们的设置恰好缺少上述增益来源**：同分布、准静态（1200 s 相干；按 116 符号/11.6 ms 推算，128 帧约 13 s，仍远小于相干时间）、线性 FIR + AWGN、无编码、CIR 已接近 oracle。文献因此预测：在当前设置下，**神经在线增量应接近零**；这与 bayes_head_diag 只增加约 1 个比特、oracle head 增益在留出帧消失一致。
7. **仍可能有头 room 的地方只有五类**：非高斯/稀疏信道结构（非线性 MMSE）、0 dB 的错误传播与可靠性建模、Q/R 或符号方差模型失配、残余同步/相位/tail 等确定性 impairment、以及 pilot 极少时的校准（但校准要转化为硬判决或解码收益）。其中只有前四类在我们的仿真中可能真实存在；**校准类增益在无编码 BPSK 下没有出口**。

**一句话判断**：当前最强的在线增益是“Bayes CIR 对 acquisition 先验到当前帧 CIR 的偏移的适应”，它本身已经是一次“分布偏移适配”，并随后验收敛而饱和；在同一波形、同一物理跟踪的冻结网络之上，神经 head 没有独立的偏移可跟踪。若继续做神经子主张，必须证明存在一个**新的、可辨识的、跨帧持续的残差方向**，否则应如实报告负结果。

---

## 1. 本报告与 v1 的分工

仓库已有 [method_survey_20261106](../method_survey_20261106/README.md)（下称 v1），覆盖了低秩贝叶斯状态空间、软 Data ELBO、数据辅助 CIR、预测式元学习、漂移门控五条线。本 v2 不重复 v1 的机制介绍，新增：

- 用 **Bayes gap / BCRB / BCJR 最优性** 三条理论，解释“为什么强基线之上没有增量”；
- 补充 **2025–2026 的负结果与“训练包络”证据**（Training Coverage、Obeed matched/shifted、Cross-site forgetting、Illusion of Progress、Realistic TTA、When TTT Fails）；
- 给出 **“失败机制 → 文献解释 → 我们的诊断 → 可检验推论”** 对照表；
- 把候选机制按“在我们设置下最可能产生可重复 BER 增量”重新排序，并给出**最小实验与 go/no-go 判据**；
- 补充公开代码/数据集/benchmark。

---

## 2. 检索与核验边界

### 2.1 检索时间与关键词

检索时间：2026-11-06。关键词覆盖：

- test-time adaptation / test-time training failure, no free lunch, negative transfer
- Bayesian Cramér–Rao bound, pilot-to-prediction, LMMSE benchmark
- deep learning channel estimation equivalent to MMSE / LMMSE
- BCJR optimal equalizer, neural equalizer cannot surpass MAP
- online fine-tuning neural receiver, distribution shift, training envelope
- hypernetwork receiver, modular Bayesian deep learning, uncertainty calibration
- learned Kalman filter, learned Q/R, KalmanNet, Split-KalmanNet
- model-based deep receiver, residual impairment, phase noise, PA nonlinearity
- LLR correction, turbo equalization, decoder gain
- continual learning neural receiver, catastrophic forgetting
- open-source neural receiver benchmark, Sionna, DeepRx

### 2.2 核验等级

| 等级 | 含义 |
|---|---|
| A | 本次实际打开全文（HTML/PDF/仓库文本）并逐段核验 |
| B | 本次只打开摘要页/记录页，未读全文 |
| C | 只读到检索摘要或二手页面，未打开原文；不用于关键结论 |
| D | 前次 v1 已全文核验，本次未重新打开；引用其已记录数值 |

> 说明：本次 `WebFetch` 对 arXiv/IEEE 域被网络策略拦截，改用只读 `curl` 抓取公开 HTML/PDF 到临时目录核验；未修改仓库。IEEE 全文多数不可达，因此部分 IEEE 论文只做到 B/C 级。

### 2.3 一个必须承认的空白

上述文献的“增益”几乎都不是相对“同源、同物理状态跟踪的 Frozen 网络”测得的。多数比较对象是：

- 同分布/偏移分布下的自身变体；
- LS/LMMSE/DFE 等传统基线；
- 联合训练 vs 在线训练 vs 元学习；
- 有编码/迭代解码的链路。

因此文献只能提供**机制、边界条件与失败模式**，不能替我们证明核心假设。我们的验收仍必须是同一信道、同一 Pilot/窗口/同步/物理跟踪协议下的配对 `BER_online - BER_frozen`，并在 0/5/10/15 dB 全部报告。

---

## 3. 为什么强物理跟踪之上没有神经增量

本节是 v2 的核心。每条给出“文献结论 → 数学直觉 → 我们的对应诊断”。

### 3.1 贝叶斯下界：可约误差是 Bayes gap，剩余是不可约后验方差

Wakayama 把 few-step TTT 写成核/线性化二次问题中的隐式贝叶斯推断。令 `P_n` 为 prompt（我们的 Adapt Pilot + 上下文），`X` 为 query（我们的 Data 位置），风险分解为：

```
Risk(M) = Bayes gap G(M) + posterior variance V
G(M) = E[(M(P_n,X) - M_Bayes(P_n,X))^2]
```

`V` 与预测器 `M` 无关，是即使有无穷算力和完美先验也无法消除的固有不确定性；**任何适配只能降低 `G(M)`**。在 Gaussian process 基准下，`G(M)` 对 spectral estimator 有精确分解：

```
E[(h_q(x) - f_Bayes(x))^2 | K, k_x] = sum_i a_i(x) * (q(lambda_i) - q_star(lambda_i))^2
a_i(x) = (tau^2 lambda_i + sigma^2) * (u_i^T k_x)^2
```

其中 `q` 是 TTT 诱导的谱滤波器，`q_star(lambda) = (lambda + lambda_star)^-1` 是 Bayes 最优收缩，`a_i(x)` 是 query 相关权重。两个条件缺一不可：

- **filter match**：`q` 要在 `a_i(x)` 大的模态上接近 `q_star`；
- **eigen-alignment**：只有与 query 特征 `k_x` 有非零内积的模态才影响 query。

由此得到三个可直接用于我们设置的推论：

1. **若基线已接近 Bayes 最优，`G(M)` 已接近 0，适配没有可约空间。** 我们的共享 Bayes CIR 在 5/10 dB 已接近 oracle CIR 的 BER：独立确认的 0/5/10/15 dB 配对增益约 `+0.037/+0.008/+0.002/+0.00026`，而 data-aided CIR 诊断中真实 CIR 相对 Frozen 的差值约 `-0.0343/-0.00705/-0.00109`（0/5/10 dB，定义为 `BER_oracle - BER_frozen`，负号即改善），两者量级相当，说明 CIR 相关的 `G` 已被吃掉；剩余主要是 `V`（后验方差/噪声）。
2. **固定更新步数/固定子空间会失败。** 原文 Proposition 6.1–6.3：固定 `T` 在测试时先验偏移下可 Bayes-suboptimal，`T -> infinity` 会过拟合；**任何固定可表示子空间都会留下不可约的 mismatch floor**，调步数不能克服子空间不匹配。[W26]
3. **“prompt 拟合好”不等于“query 好”。** 原文实验里 Trace-TopK 优先拟合 prompt 主特征，prompt-fit MSE 改善，但 query 改善不如 query-aware 选择；作者明确写“simply fitting the model to the prompt is not sufficient”。[W26]

**对应我们的失败**：v16–v22 的低维子空间方向来自 Adapt Pilot 的 Fisher/Jacobian（prompt），门控用 Reward Pilot（仍是 prompt），评估用 Data（query）。这正是 `a_i(x)` 与 `(u_i^T k_x)^2` 不匹配的情形。v6 gate-alignment 的会话内 Spearman 在 0/5/10/15 dB 约 `+0.11/-0.14/-0.34/-0.60`，5–15 dB 为负，说明 Reward Pilot 的动作排序与 Data 风险方向相反；这不是阈值问题，而是 query 错配。

### 3.2 估计侧：LMMSE = BCRB，神经架构已贴界

Lagona、Bockelmann、Dekorsy 对频率选择性 OFDM 的严格因果 pilot-to-prediction 推导了闭式 LMMSE 基准。在 proper complex Gaussian、WSSUS、pilot 观测加性高斯等标准假设下：

- **该 LMMSE 基准等于 Bayesian Cramér–Rao 界（BCRB）**；
- 误差可加性分解为：
  - **不可约时间预测误差**：由 Doppler/信道演化决定，与 SNR 无关；
  - **pilot 诱导的估计误差**：由 pilot 密度、SNR、时间相关决定，有闭式特征值表达。
- 五个神经架构（Transformer、SSM/Mamba-Hybrid、LSTM、Autoformer、Crossformer）中，**Transformer 与 SSM 在 COST 259 多数 Doppler 区间紧贴基准 5% 以内**；LSTM/Autoformer 差距更大；Crossformer 在有限 SoS 信道（非高斯）可低于基准，但在 3GPP TDL-A 上仍有正差距。
- 原文结论：**“further gains depend primarily on pilot density, signal-to-noise ratio, or prediction horizons.”**[L26]

**对应我们的设置**：116 tap、224 个 Adapt Pilot（约 1.9× tap 数）、准静态信道。CIR 估计误差的主要项已经被 Bayes CIR 的后验吃掉；在 5/10 dB，Bayes CIR 达到 oracle CIR 的 BER 水平。BCRB 视角下，**剩下的不是“网络不够好”，而是“观测信息已经用尽”**。15 dB 的 Adapt 梯度约 `2.6e-6`（v6 重建诊断 `3.51e-6`，v14 筛查 `7.56e-8`）也说明 Fisher 信息趋零。

### 3.3 线性高斯模型：DL 最多逼近 MMSE

Hu、Gao、Zhang、Jin、Li 证明：ReLU DNN 在数学上等价于分段线性函数，可万能逼近一大类函数；DL 信道估计**不依赖特定信号模型，可在多种场景逼近 MMSE**，因此“优于或至少可比”传统估计器，取决于信道类型。[H20]

关键限定：

- 在**线性**信号模型下，DL 估计器只是接近 LMMSE，**不能超过 LMMSE**；
- 显著超过 LMMSE 的情形来自**非线性信号模型**（如量化、非线性前端、非高斯先验）；
- DL 对训练数据质量敏感；真实信道比训练分布更宽时会退化，而 LMMSE 对统计失配相对稳健。

**对应我们的设置**：信道是线性 FIR + AWGN，没有 PA/量化/相位噪声等非线性。BPSK 星座本身是非高斯的，所以非线性 MMSE 理论上可优于 LMMSE；但我们的离线神经均衡器已经承担了这部分非线性收益。**在线 head 要再有增量，必须找到离线网络没有吃掉的非线性/非高斯结构**，而不是重复 LMMSE 已经做的事。

### 3.4 完美 CSI / 匹配分布：神经只能匹配 BCJR，不能超过

Rozenfeld、Raphaeli、Bialer 的结论非常直接：[R24]

- **BCJR 是 BER 最优均衡器**；在已知 CSI 下，没有均衡器能超过它；
- 小参数 NN 均衡器容易收敛到**次优局部极小，连 LMMSE 都超不过**；
- 用 **LMMSE 初始化 + 渐进学习** 后，小 NN 可以超过 LMMSE，并缩小与 MAP/BCJR 的差距，但**仍不能超过 BCJR**；
- 在 turbo 均衡中，BCJR 的 LLR 质量通常优于 LMMSE，与 SNR 和迭代次数无关。

相邻证据（B/C 级）：

- SICNet / DeepIM-SIC：完美 CSI 下经典 SIC/模型检测最好，神经接收机只是接近；增益出现在**不完美 CSI** 的高 SNR 区。[C]
- TDMR 的 DNN APP 检测器：相对线性均衡器有增益，但作者明确说线性均衡器本来就显著差于 2D-BCJR/Viterbi，增益来自 media noise/状态爆炸，不是超过理想 BCJR。[C]

**对应我们的设置**：116 tap 的全状态 BCJR 不可行（`2^116`），所以“实际最优”是 reduced-state BCJR / turbo / 长记忆 DFE。我们的离线神经均衡器是在逼近这个最优；**在线 head 的对手不是 Frozen，而是“同一物理跟踪下的 reduced-state BCJR”**。如果离线网络已经接近它，head 就没有空间。当前 oracle head 增益在留出帧消失，说明“同帧最优 head”本身不可跨帧迁移。

### 3.5 在线适配的增益主要是“分布偏移恢复增益”

这是解释“为什么匹配分布下没有增量”的最强经验证据。

**Obeed & Jian（2026，DMRS 持续学习）**：[O26]

- 匹配训练分布时，神经接收机显著优于不完美 CSI 的 LMMSE，并**接近不可实现的完美 CSI LMMSE**；
- 分布偏移（时延扩展改为 400–410 ns）后，固定神经接收机 BER 急剧恶化，高 SNR 尤其明显；
- 在线微调 24k 样本后 BER 明显改善，50k 样本后**恢复到原始性能**；
- 在慢/快连续漂移下（例如每 1000 mini-batch 改变 20 ns），在线模型 BER 保持在 `2e-3` 以下并随适配周期继续改善，固定模型升到 `5e-2`；
- 关键：**它恢复的是“偏移造成的损失”，不是在匹配分布上再赚一层**。

**Training Coverage（2026，5G NR PUSCH）**：[T26]

- 基线包括 **genie LMMSE**（每个测试场景的经验统计）与 practical LMMSE（训练混合的固定统计）；
- **训练包络内，两种学习接收机都只是匹配 genie 基线**；
- 包络外：2000 ns RMS 时延扩展时，学习接收机比 genie 少需 `0.90 dB`，practical LMMSE 根本达不到 10% BLER；单 DMRS、20 m/s 时少需 `4.08 dB`，30 m/s 时只有神经接收机能工作；
- 把训练时延扩展从 30–300 ns 扩到 30–1000 ns，可把 2000 ns 处的 `-5.97 dB` 劣势变成 `+0.90 dB` 优势，代价是 45 m/s 内最多 `0.1 dB`；
- 同一训练范围下，**保留 LS/LMMSE 前端、只学习 LLR 残差修正的 hybrid 接收机**，以 `3.9×` 更少参数/FLOPs 匹配完整神经接收机到 `0.05 dB` 以内；
- 还记录了一个**基线陷阱**：基于 DMRS 的 CFO 补偿会同时移除强 LOS 径的 Doppler shift，破坏 LMMSE 相关模型，损失 `0.84 dB`，从而制造“学习接收机的虚假增益”。

**对应我们的设置**：同分布、准静态；按 116 符号/11.6 ms 推算，128 帧约 13 s，远小于 1200 s 相干时间，**没有可恢复的分布偏移**。Bayes CIR 的增益来自“acquisition 先验 → 当前帧 CIR”的偏移；当后验收敛，这个偏移也被吃掉，于是出现“先增长后饱和”。神经 head 没有第二个独立偏移可跟踪。

### 3.6 子空间/查询错配：为什么“Pilot 梯度对齐”仍然不够

除了 3.1 的 Wakayama 理论，v1 已记录的低秩贝叶斯状态空间（Gusakov 等）也隐含同一问题：它更新的是**模块参数**，观测是 streaming Pilot；它能修复“高 SNR 梯度消失”，但**不能创造 Pilot 中不存在的信息**。[G26]

我们的诊断正好落在两者之间：

- v6 梯度重建：Adapt/Data 余弦在 0/5/10/15 dB 约 `0.408/0.236/0.224/0.169`；15 dB Adapt 梯度 `3.51e-6`，Data 梯度 `1.54e-3`；
- v14 只读筛查：15 dB Adapt 梯度 `7.56e-8`；
- v20/v21/v22：Pilot 上下文 adapter 在 0/5 dB 有小正点估计，但换 seed 不复现；
- v16/v17：12 维子空间 + 梯度传输矩阵仍与 Frozen 相同；oracle 诊断显示候选只翻转 1 个错误；
- v24–v27：GN/近端元学习没有稳定收益，5 dB Adapt/Data 平均余弦约 `-0.018`，GN 候选平均增加 Data BCE `7.93e-5`。

用 Wakayama 的语言：**我们的更新方向是 prompt-aligned，不是 query-aligned；固定子空间的 mismatch floor 已经大于可约增益**。这不是再换一个损失函数能解决的。

### 3.7 校准收益需要下游解码器；无编码时没有出口

Raviv 等的模块化贝叶斯深度接收机（modular Bayesian DL）给出一个非常关键的“负结果式”细节：[B25]

- 在 `Q=1`（单模块）时，**frequentist 与 modular Bayesian DeepSIC 的 SER 相同**，贝叶斯只改善 ECE（校准）；
- BER 的改善来自**后续解码器**利用更可靠的软输出；
- 多模块 `Q>1` 时，校准才同时降低 SER 和 ECE，并进一步降低 BER；
- 总体 BER 增益可达 `1.6 dB`，但增益路径是“校准 → 解码”。

相邻证据（B/C 级）：

- LLR correction：DNN 纠正过/欠自信 LLR，在 LDPC 迭代检测中 `0.9 dB @ BLER 1e-3`，复杂度 +16%。[C]
- 无编码 BPSK 下，**单调的 LLR 缩放不改变硬判决符号**；只有改变判决边界（非单调修正/新检测器）才可能改变 BER。因此“不确定性条件化”“贝叶斯集成”“温度校准”这类机制，在我们的当前指标下**没有直接出口**。

**对应我们的设置**：无 FEC、无迭代解码、BPSK 硬判决。这直接排除了 v1 中 M2 的“校准收益”路径，也解释了为什么 modular Bayesian 类方法即使实现，也可能只改 ECE 不改 BER。

### 3.8 小参数网络局部极小、负迁移与超参敏感

- **小参数 NN 均衡器**：有限参数下容易落入次优局部极小，**连 LMMSE 都超不过**；LMMSE 初始化 + 渐进学习是关键。[R24] 我们的 2/4/8/12 维 head 正是这个 regime；v24–v27 的 GN/近端失败与“局部极小/方向错误”一致。
- **负迁移/遗忘**：跨站点 OTA 神经接收机微调在新站点 `10^-2` 处增益 `0.6 dB`，但让其他站点灵敏度下降约 `0.1 dB`；联合训练没有这个损失。[X26]
- **TTA 收益被高估**：TTA-VLM 基准发现现有 TTA 相对早期方法增益有限、与训练时微调兼容性差、精度增益常以校准/鲁棒性为代价。[I25]
- **现实假设下 TTA 会伤害零样本性能**：当 batch 内类别数可变、在线流非 i.i.d. 时，当前 transductive/TTA 方法会系统性损害 CLIP 的零样本鲁棒性。[RT25]
- **TTT 在严重偏移下会退化**；TTT++ 用特征对齐缓解，说明“自监督 TTT 不是默认安全”。[TT21, C]
- **TTT 超参极敏感**：学习率、步数、weight decay、momentum、梯度归一化都没有通用配方；SGD 反而优于 Adam。[WTF26, C]

**对应我们的设置**：v12/v13 的 Pilot-only 风险门控失败、v14 配对外层目标失败、v15 错误条件化失败，都与“小数据/小参数/查询错配/超参敏感”一致。继续在同一结构上扩 seed 矩阵，文献不支持。

### 3.9 失败机制对照表

| 失败机制 | 文献证据 | 我们的对应诊断 | 可检验推论 |
|---|---|---|---|
| 基线已近 Bayes 最优，剩余是不可约后验方差 | [W26] Bayes gap 分解；[L26] BCRB | Bayes CIR 在 5/10 dB 达 oracle CIR 水平；共享 Bayes 后 head 仅 +1 bit | 若 oracle CIR 与 Bayes CIR 的 BER 差已 < 噪声，任何 head 更新都不应期待正增益 |
| 估计侧已贴 BCRB | [L26] Transformer/SSM 在 5% 内 | 224 Pilot / 116 tap，准静态 | 增大 Pilot 或 SNR 才可能提升；同资源下无空间 |
| 线性高斯下 DL≈MMSE | [H20] | 信道为线性 FIR + AWGN | 除非存在非线性/非高斯结构，否则不能超过 LMMSE/MMSE |
| 完美 CSI 下不能超过 BCJR | [R24]；SICNet/DeepIM-SIC [C] | 离线网络已逼近长记忆最优；oracle head 增益不跨帧 | 在线 head 的对手是 reduced-state BCJR，不是 Frozen |
| 适配增益=分布偏移恢复 | [O26] matched vs shifted；[T26] inside vs outside envelope | 同分布、准静态、128 帧 << 相干时间 | 没有 shift 可恢复；Bayes CIR 的增益本身是 shift 适应并会饱和 |
| 子空间/query 错配 | [W26] prompt fit ≠ query；固定子空间 floor | v16–v22 用 Adapt Fisher 选方向、Reward 门控、Data 评估 | 必须用 query-aware 子空间；否则换损失函数无用 |
| 校准需要解码器 | [B25] Q=1 SER 不变；[C] LLR correction 0.9 dB with LDPC | 无编码 BPSK 硬判决 | 校准/集成类机制不改变 BER |
| 小参数局部极小/负迁移/超参敏感 | [R24]；[X26]；[I25]；[RT25]；[WTF26] | 2–65 维 head 的 GN/prox/Adam 均不稳定 | 初始化与子空间比更新规则更关键；扩 seed 无益 |
| 高 SNR Pilot 梯度消失 | Fisher `p(1-p) -> 0`；[G26] EKF/自然梯度可修复但需信息 | 15 dB Adapt 梯度 1e-6–1e-8；Adapt/Data 余弦近零 | 若信息不存在，EKF 也无效；先做可辨识性审计 |

---

## 4. 文献总表

> 列说明：**标签**=是否用 Data/真实标签；**防偏**=如何避免错误传播/确认偏置；**增益**=原文报告的相对强基线结果；**增长/饱和**=是否报告随帧/块增长的证据；**核验**=A/B/C/D。

| # | 来源 | 机制（更新什么/观测/损失） | 标签 | 防偏 | 相对强基线增益 | 增长/饱和 | 计算/延迟 | 局限 | 核验 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wakayama, *A Decision-Theoretic View of TTT*, ICML 2026, PMLR 306:124279–124317; arXiv:2606.15569 [W26] | 不提出新接收机；把 few-step TTT 解释为核/线性化二次问题中的隐式贝叶斯推断，给出 spectral Bayes-gap 分解与 query-aware 子空间选择 | 否（理论） | 用 prompt evidence 选步数，PAC-Bayes 防过拟合；query-aligned 子空间 | 无 BER；解释 TTT 不稳定/失败 | 局部 few-step；不覆盖全参数/长轨迹 | N/A | 局部线性化；高斯过程基准；非通信专用 | A |
| 2 | Lagona, Bockelmann, Dekorsy, *On the Bayesian Performance Limits of Pilot-to-Prediction*, IEEE OJ-COMS 2026, DOI:10.1109/OJCOMS.2026.3721910 [L26] | 不更新参数；推导严格因果 P2P 的闭式 LMMSE 基准 = BCRB，分解预测/估计误差；基准五种神经架构 | 否（基准） | 用 genie 统计做公平基线；分离不可约误差 | Transformer/SSM 紧贴基准 5%；Crossformer 在有限 SoS 可低于基准 | 未报告帧增长；给出误差分解 | Transformer/SSM 推理；未给统一延迟 | 线性 WSSUS；SISO；OFDM | A |
| 3 | Hu, Gao, Zhang, Jin, Li, *Deep Learning for MIMO Channel Estimation: Interpretation, Performance, and Comparison*, arXiv:1911.01918 [H20] | 离线监督训练 DNN 做信道估计；分析 ReLU 分段线性与 MMSE 逼近 | 是（离线标签） | 结构/训练数据分析；非在线 | 线性模型下接近 LMMSE，不显著超过；非线性模型下可超过 | 未报告 | 未给统一延迟 | 线性模型无理论增益；训练分布敏感 | B |
| 4 | Obeed & Jian, *Learning During Detection: Continual Learning for Neural OFDM Receivers via DMRS*, arXiv:2602.20361 [O26] | 复用 DMRS 同时解调与在线微调；三种 pilot 设计；并行/前向复用两种架构 | 是（DMRS 标签） | 随机化 pilot 防过拟合；并行架构避免服务中断；分布偏移检测触发 | 匹配分布接近完美 CSI LMMSE；连续漂移下在线保持 <2e-3、固定升到 5e-2；突然偏移后在线微调恢复到接近原始性能 | 随适配周期持续改善后收敛 | 分析前向/反向延迟；未给统一 ms | 增益依赖分布偏移；OFDM/DeepRx/CoNet | A |
| 5 | *Training Coverage, Not Architecture, Decides When Learned 5G NR Receivers Help*, Zenodo 22994791 (2026, preprint) [T26] | 全卷积神经接收机 vs 保留 LS/LMMSE 前端、学习 LLR 残差修正的 hybrid；标准 5G NR PUSCH | 是（离线训练） | genie/practical LMMSE 双基线；训练包络消融；记录 CFO 基线陷阱 | 包络内只匹配 genie；包络外 0.90–4.08 dB；hybrid 以 3.9× 更少参数匹配 0.05 dB 内 | 未报告帧增长；重点是训练覆盖 | 3.9× 更少参数/FLOPs | 预印本未同行评审；OFDM；需要训练覆盖 | B |
| 6 | Raviv & Shlezinger, *Modular Hypernetworks for Scalable and Adaptive Deep MIMO Receivers*, IEEE 2025; arXiv:2408.11920 [MH25] | 离线训练 hypernetwork，按瞬时信道/用户数生成 DeepSIC 模块权重；在线只前向 | 是（离线训练） | 模块化；不更新在线参数；用户嵌入 | 优于 joint training，接近 online training 的 SER，复杂度显著更低 | 按 block 自适应；未报告帧增长 | 推理时生成权重；解析复杂度低于在线训练 | 不能超过 online training；MIMO；需要离线覆盖 | A |
| 7 | Raviv, Park, Simeone, Shlezinger, *Uncertainty-Aware and Reliable Neural MIMO Receivers via Modular Bayesian DL*, IEEE TVT 74(11):17637–17651, 2025; arXiv:2302.02436 [B25] | 对 DeepSIC/wBP 每个模块做贝叶斯学习（ELBO/ensemble），校准内部软输出 | 是（Pilot 标签） | 贝叶斯后验/集成；模块级校准；顺序训练 | QPSK 最高 0.2 dB、8-PSK 最高 1 dB SER；整体 BER 最高 1.6 dB（需解码器） | 未报告帧增长；强调 scarce data | 集成 J 次前向；训练更慢 | Q=1 时 SER 不变，仅 ECE；需要下游解码器；先验/似然失配时不可靠 | A |
| 8 | Raviv, Park, Simeone, Eldar, Shlezinger, *Online Meta-Learning for Hybrid Model-Based Deep Receivers*, IEEE TWC 22(10):6415–6431, 2023 [R23] | 预测式元学习：内循环当前帧 Pilot + FEC 自监督；外循环下一帧 query；学习初始化/超参 | 否（用 decoder 正确性/重编码） | 只在 decoder 报正确时用该块；模块化只更新动态模块 | 相对 self-supervision/joint learning 最高 2.5 dB coded BER | 报告 BER vs block index；未报告饱和统计 | 元训练成本高；在线每块更新 | 依赖 FEC/CRC；我们无编码 | D |
| 9 | Gusakov, Simeone, Routtenberg, Shlezinger, *Online Learning of Modular Bayesian Deep Receivers*, IEEE TSP 74:2192–2207, 2026; arXiv:2511.06045 [G26] | 把模块参数当隐状态，单步 EKF/自然梯度；Vd-EKF/Lo-Fi/CM-EKF | 否（streaming Pilot） | 参数后验协方差 + 过程噪声；模块化局部更新 | 相对 SGD/BBB/BONG-EF，CM-EKF BER 最低；线性旋转信道 6 snapshot 达最优 0.2% 内（NLMS ~12、SGD ~40） | 快速同步后进入跟踪；未报告“增益随帧增长”曲线 | CM-EKF 2.942 ms/sample；Lo-Fi 0.356；Vd-EKF 0.103；GD-10 0.268 | 需要模块足够小；全协方差 O(P^2)；Pilot-only | D |
| 10 | Burshtein & Bery, *Semi-Supervised Channel Equalization Using VAEs*, IEEE TWC 23(12):19681–19695, 2024; arXiv:2309.11841 [BB24] | 半监督 VAE：Pilot 标签 + payload ELBO + 星座先验 KL；对 payload 不取硬标签 | 否（payload 无标签） | 对星座后验边缘化；KL 到先验 | 显著减少所需 Pilot；优于使用当前+历史 Pilot 的 meta-learning；约 512 符号饱和 | 约 512 符号后饱和 | VAE 训练/推断高于线性均衡 | 无记忆/短记忆非线性；长 ISI 留作未来 | D |
| 11 | Caciularu, *Geometric Inductive Biases for Semi-Supervised Equalization: CAT*, arXiv:2609.33695 (2026 preprint) [CAT26] | 每 block test-time training：Pilot 标签 + payload 半监督 ELBO；星座 token 注入 attention | 否（payload 无标签） | 半监督 ELBO + 星座先验；软后验；LLR 校准 | 5G NR LDPC、BLER 1e-2 时比 vanilla Transformer +2.5 dB；ECE 0.012 vs 0.089 | 每 block 独立训练，不跨帧累积；5000 步/block；meta-init 后 50 步差 2% | 5000 步/block；step time +17–28%（16-QAM） | 预印本；需要最少 Pilot；仅仿真 | D |
| 12 | Shi, Qi, Chen, Wang, *SIRIUS: Unsupervised Online Channel Estimation via INR*, arXiv:2605.10213 (2026 preprint) [SIR26] | 每 slot 用 SIREN/INR 拟合时频连续信道；物理 ICI 重建 + 相邻正则；Pilot + 高置信伪导频 | 否（Pilot + 伪导频） | 物理重建 + 置信阈值 + 伪导频降权 | 接近最优链路可靠性，优于 LS/robust LMMSE；OOD 优于监督 hybrid | 每 slot 独立；高 SNR 有 NMSE floor | 0.029M 参数；0.7 s/slot | 延迟大；一阶 ICI 截断；伪导频误差 | D |
| 13 | Rozenfeld, Raphaeli, Bialer, *Enhancing LMMSE Performance with Modest Complexity Increase via NN Equalizers*, arXiv:2411.01517 (2024 preprint) [R24] | 小参数 NN 均衡器；用 LMMSE 滤波器初始化 + 渐进学习；输出 LLR | 是（离线监督，目标 MAP LLR） | LMMSE 初始化避免局部极小；渐进扩展感受野 | 超过 LMMSE，缩小与 BCJR 的差距；但不能超过 BCJR；有限参数不加初始化会输给 LMMSE | 性能随 K 增长到 `M_h+2` 后饱和 | 复杂度约 `K × O(LMMSE)` | 不能超过 BCJR；局部极小；需要 CSI | A |
| 14 | Pihlajasalo et al., *HybridDeepRx: Deep Learning Receiver for High-EVM Signals*, IEEE PIMRC 2021; arXiv:2106.16079 [HDR21] | 时域+频域 CNN 接收机，处理 PA 非线性导致的高 EVM | 是（离线训练） | 时域层建模非线性；频域层检测 | 在高 EVM 下明显优于经典线性接收机和现有 ML；可把 PA 推更深饱和 | 未报告帧增长 | 未给统一延迟 | 增益来自非线性/PA；不是在线适配 | B |
| 15 | Shen, Li, Sun, Xu, Zhang, Zhao, *Learnable Semi-Blind Receiver Design for Phase Noise Impaired OFDM*, IEEE TCOM 73(12):13846–13862, 2025 [PN25] | JCPD：联合信道估计、PN 补偿、数据检测；ADMM+MM 展开为模型驱动网络 | 是（半盲/Pilot） | 模型结构 + 展开；PN 物理模型 | 报告优于传统 JCPD/线性接收机（具体 dB 未逐表核验） | 未核验 | 每层约 FFT 复杂度 | 相位噪声专用；OFDM；未核验全文 | C |
| 16 | *Low-Complexity and Low-Energy RNN-Enhanced Kalman Filter for Channel Estimation*, IEEE 2025, doc 11397385 [SKF25] | Smart Kalman Filter：两个轻量 NN 动态调整 KF 的 Q/R；用 Pilot | 是（Pilot） | 保持 KF 状态空间；可解释；量化 | 相对 LS 低 SNR 最高 5 dB、高 SNR 3 dB | 未报告帧增长 | ARM 53.7 Mbps；FPGA 8–20 Mbps | 基线是 LS，不是强 KF；未核验全文 | C |
| 17 | *Deep Learning Aided LLR Correction Improves Iterative MIMO Receivers*, IEEE TVT 73(2):2045–2060, 2024, DOI:10.1109/TVT.2023.3312029 [LLR24] | DNN 检测并纠正过/欠自信 LLR；用于 SOCA MIMO + LDPC 迭代检测 | 是（离线训练） | LLR 可靠性一致性；EXIT 分析 | `0.9 dB @ BLER 1e-3`，复杂度 +16% | 未报告帧增长 | +16% 复杂度 | 需要 LDPC/迭代解码；MIMO | C |
| 18 | Luostari, Korpi, Holma, Tirkkonen, *Cross-Site Transfer and Spatial Forgetting in OTA-Trained Neural Receivers*, arXiv:2610.02577 (2026) [X26] | 跨站点迁移/微调；8 站点 5.88 GHz 测量；四种训练制度 | 是（测量标签） | 联合训练 vs 微调；多 seed | 新站点 `10^-2` 处微调 +0.6 dB；但其他站点 -0.1 dB（空间遗忘） | 未报告帧增长 | 未给统一延迟 | 遗忘/负迁移；OTA 特定 | B |
| 19 | *The Illusion of Progress? A Critical Look at TTA for VLMs*, arXiv:2506.24000 (NeurIPS 2025 D&B) [I25] | 基准评估 8 种 episodic + 7 种 online TTA | 否（TTA） | 统一协议、多轴评估（校准/OOD/对抗） | 现有 TTA 相对早期方法增益有限；与训练时微调兼容性差；精度增益常牺牲可信度 | 不适用 | 不适用 | VLM 领域；非通信 | A |
| 20 | Zanella et al., *Realistic Test-Time Adaptation of Vision-Language Models*, CVPR 2025; arXiv:2501.03729 [RT25] | 现实假设（可变有效类别数、非 i.i.d. 在线流）下评估 transductive/TTA | 否（TTA） | 统计锚点正则（StatA） | 当前 TTA 系统性损害零样本鲁棒性 | 不适用 | 不适用 | VLM 领域；非通信 | B |
| 21 | Ha, Lee, Jeon, Jeon, *Online Data Generation for MIMO-OFDM Channel Denoising: Transfer Learning vs Meta Learning*, arXiv:2508.09751 (2025) [ODG25] | 用 data-aided 估计生成在线训练数据（虚拟 DMRS）；TL 微调 vs MAML | 否（虚拟导频） | 相邻检测符号作虚拟参考；推导最优邻域大小；可靠符号选择 | 报告优于传统信道估计（具体 dB 未逐表核验） | 未报告帧增长 | 在线微调/MAML 少步 | 信道去噪而非均衡；需要可靠检测 | A |
| 22 | Xiao et al., *Spike-Kal: A Spiking Neuron Network Assisted Kalman Filter*, arXiv:2504.12703 (2025) [SK25] | SNN 直接从观测逼近最优 Kalman 增益，替代矩阵运算 | 是（离线训练） | 保持 KF 结构；对时变噪声自适应 | 平均误差相对其他方法降低 18%–65% | 未报告帧增长 | SNN 低功耗 | 跟踪任务；非通信；未核验全文 | B |

> 读法：第 1/2/3/13 行是“边界/负结果”而非“增益方法”；第 4/5/18 行是“偏移/迁移”证据；第 6/7/8/9/10/11/12/21 行是“适配机制”；第 14/15/16/17/22 行是“物理/校准/模型失配”机制。不要把表中的 dB/MSE 直接当作我们核心假设的证据。

---

## 5. 仍有 head room 的作用点/信息源

| 作用点/信息源 | 文献证据 | 我们设置是否具备 | 判断 |
|---|---|---|---|
| 分布偏移（时延扩展/Doppler/SNR 漂移） | [T26] 包络外 0.90–4.08 dB；[O26] 偏移后恢复 5e-2→<2e-3 | 否：同分布、准静态、128 帧 << 1200 s 相干 | 不具备；且 AGENTS 禁止改信道制造偏移 |
| 硬件非线性（PA/IQ/量化/相位噪声） | [HDR21] 高 EVM 下明显优于线性；[PN25] PN 专用 | 否：线性 FIR + AWGN | 不具备（除非仿真显式含非线性） |
| 非高斯/稀疏信道结构 | [H20] 非线性模型可超 LMMSE；[L26] Crossformer 在有限 SoS 可低于 BCRB | 部分：EME 长回波可能稀疏；但 Bayes CIR 已用高斯先验 | **可能**：学习稀疏/重尾先验，或非线性 MMSE |
| 稀缺 Pilot / 低 SNR | [B25] 384 Pilot 下 0.2–1 dB；[CAT26] +2.5 dB BLER；[BB24] 512 符号饱和 | 部分：Adapt=224，0 dB 稀缺；但离线训练数据充足 | **可能**：0 dB/短 Pilot 的校准与可靠性 |
| 编码/迭代解码 | [B25] BER 1.6 dB；[LLR24] 0.9 dB | 否：无 FEC/迭代解码 | 不具备；校准类机制无出口 |
| 数据辅助虚拟导频/软似然 | [ODG25]；[SIR26]；[BB24]；[CAT26] | 部分：有 768 Data 符号；但 CIR 已近 oracle | **可能但信息重复**：需证明 Data 提供 Pilot 之外的信息 |
| Q/R 或符号方差模型失配 | [SKF25]；Split-KalmanNet [C]；[G26] | 部分：Bayes CIR 用固定 Q/R；0 dB H3 不通过 | **可能**：0 dB 错误传播/可靠性建模 |
| 0 dB 错误传播/非线性 MMSE | [R24] BCJR 差距；[H20] 非线性 | 是：0 dB BER ~17–23% | **可能**：但离线网络已承担大部分 |
| 残余同步/相位/timing/tail | [HDR21]；P0 tail 对齐 | 部分：tail 已修；相位由 CIR 吸收 | **低–可能**：需 oracle 残差诊断 |
| 不确定性/校准 | [B25]；FiLM-Ensemble [C] | 是：Bayes 后验协方差可用 | **低**：无解码器时硬 BER 不变；但可作为上下文 |

**结论**：在我们的仿真中，唯一“真实存在且未被 Bayes CIR 吃掉”的头 room 是：

1. **0 dB 的错误传播/可靠性建模**（H3 不通过、BER 仍高）；
2. **非高斯/稀疏先验或非线性 MMSE 的残余**（需证明离线网络没有吃干净）；
3. **Q/R/符号方差模型失配**（固定参数在 0 dB 可能次优）；
4. **可能的残余确定性 impairment**（需 oracle 诊断，不能假设存在）。

其余文献增益来源（偏移、非线性硬件、编码、稀缺 Pilot）在我们当前设置中要么不存在，要么没有出口。

---

## 6. 候选机制排序

> 排序标准：在“共享 Bayes CIR、同分布、无编码、线性 FIR、0/5/10/15 dB”下，**最可能产生可重复配对 BER 增量**。
> 先验概率是主观的，基于第 3 节理论与第 5 节头 room；所有候选都必须先过“只读诊断”，再训练，再独立 seed 复核。
> 共同约束：Data 标签只用于离线训练/选择/事后评分；在线只用 Adapt Pilot 更新、Reward Pilot 门控；不改信道/Pilot/窗口；Frozen 与 Online 共享同一 Bayes CIR、tail、波形。

### 排名总表

| 排名 | 机制 | 目标头 room | 先验 | 最小实验 | go/no-go | 主要失败模式 |
|---|---|---|---|---|---|---|
| 1 | **Query-aligned 不确定性条件化低秩残差（FiLM/Adapter）** | 子空间/query 错配 + 后验不确定性 | 中低 | 先做 query-vs-prompt 子空间捕获率诊断，再训 2–4 维 context adapter | 至少一个 SNR 的配对 ΔBER 区间不跨零且 3 seed 同向；或 query 子空间捕获 oracle 增益 >70% | oracle 方向不跨帧；context 特征无预测力；r 太小 |
| 2 | **软 Data 似然（ELBO）+ 可靠符号门控，只更新 2–4 维 head** | Data 信息源 / Pilot 梯度消失 | 低中 | 逐帧 ELBO 梯度范数与 oracle Data 梯度余弦；固定半径步进；3 seed × 16 帧 | cos > 0.3 且范数高于噪声底；≥2/3 seed BER 改善 | 残差被不可约噪声主导；ELBO 自洽不等于真值；确认偏置 |
| 3 | **学习式可靠性/符号方差模型（R_t 或 Q/R 的神经修正）** | 0 dB 错误传播/Q-R 失配 | 低中 | 0 dB 5 seed × 128 帧；固定 R vs 学习 R；16 帧分箱斜率 | 0 dB 斜率转正且 BER 增益 > 0；5/10/15 不退化 | 噪声本身白高斯；误差相关性不可辨识；Bayes CIR 已近 oracle |
| 4 | **物理参数化残差 impairment（相位/timing/tail/额外 tap）** | 残余确定性模型误差 | 低 | oracle 逐帧拟合 θ/timing/tail，测 BER 上界；若有增益再学 Pilot 预测器 | oracle 增益 > 0 且 Pilot 预测器恢复 >50% | 仿真无此 impairment；tail 已修；相位被 CIR 吸收 |
| 5 | **Bayes 后验条件化 hypernetwork（无在线梯度，逐帧生成低秩权重）** | 梯度消失/局部极小 | 低 | 离线训练 hypernetwork，2 新 session 评估；与 EKF adapter/Frozen 比 | 至少一个 SNR 正增益，且延迟可接受 | 不能超过 teacher；teacher 本身 ~0；离线覆盖不足 |
| 6 | **学习式优化器/预条件（learned step size/preconditioner）** | 更新规则/局部极小 | 低 | 离线 replay 现有轨迹，比较每步 Data-BCE 下降；再在线 3 seed | 每步下降 > 固定 GN 20% 且稳定 | 没有信号可学；过拟合；v24–v27 已负 |
| 7 | **贝叶斯集成/不确定性校准** | 校准 | 很低 | 测 ECE + BER；无编码下预期 BER 不变 | BER 改变才算；否则只作诊断 | 单调校准不改硬判决；无解码器无出口 |

**排序理由**：

- 第 1 名是唯一**直接针对我们已诊断失败原因**（prompt-aligned 子空间、Reward/Data 方向不一致）且**引入新信息**（Bayes 后验协方差/innovation）的机制；文献 [W26] 给出了 query-alignment 的理论与可操作选择准则，[B25] 证明不确定性校准在 scarce data 下有用（但需要解码器，故只能作为上下文而非校准输出）。
- 第 2 名是唯一**独立于 Pilot 的信息源**（768 Data 符号的软边缘化），但我们的单步 ELBO 诊断已偏负，所以先验低于第 1。
- 第 3 名针对 0 dB 的 H3 失败与固定 Q/R；但 Bayes CIR 已近 oracle，空间可能很小。
- 第 4 名只有在 oracle 诊断证明存在残余 impairment 时才值得做；否则是空转。
- 第 5 名受“不能超过 teacher”限制；第 6/7 名先验很低。

---

### 6.1 M1：Query-aligned 不确定性条件化低秩残差

**为什么可能在我们“Bayes CIR 已吃掉 CIR 头 room”的情况下仍有增量**

- Bayes CIR 吃掉了 **CIR 均值** 的头 room，但没有把 **CIR 后验不确定性** 作为神经输入。当前 head 看到的是点估计 `h_hat`，不知道哪些 tap 不可靠、哪些符号的软输出不可信。
- Wakayama 的结论是：适配要有效，更新方向必须与 **query 相关特征方向** 对齐；我们过去的子空间来自 Adapt Pilot 的 Fisher（prompt），因此即使 Pilot 梯度对齐，也可能落在 query 的零空间。[W26]
- 后验协方差/innovation 是**合法的在线信息**（只由 Pilot 与已接收波形得到），不违反标签隔离；它可以作为 FiLM/Adapter 的 context，决定“在哪些方向、以多大步长”更新。
- 如果 query-aligned 方向存在但被 Pilot 子空间漏掉，这是唯一可能把“已吃掉 CIR 头 room”之后剩余的非线性/可靠性结构利用起来的路径。

**参数化（草图）**

```python
# 冻结主干 + 共享 Bayes CIR；在线只更新 r 维系数 a_t
z, mu, var = frozen_equalizer(y, h_bayes, Sigma_h)   # h_bayes, Sigma_h 来自 Bayes CIR
c_t = context(
    diag(Sigma_h),                 # 逐 tap 后验方差
    innovation_norm,               # ||y - H mu||^2
    mean_entropy,                  # Data 软输出平均熵
    pilot_residual,                # Adapt Pilot 拟合残差
    snr_hat, frame_age
)
# context -> 有界低秩基 B(c) 和步长/门控
B_t = B_phi(c_t)                   # d x r, r=2/4
a_t = a_{t-1} + clip( W_phi(c_t) @ (b_pilot - p_pilot), delta )
delta_z = B_t @ a_t
```

- `B_phi`、`W_phi` 是小的 context 网络（2 层 MLP），输出有界；`a_t` 是唯一在线神经参数。
- 若要用 LoRA 形式：对冻结线性层加 `ΔW = B A`，只训练/更新 `A` 的 r 维行（或等价的低秩系数），`B` 由 context 生成；零初始化 `A=0` 精确保留 Frozen 输出。
- 零上下文时 `B_t a = 0`，精确保留 Frozen 输出；`a_0 = 0`。
- 离线外循环：内循环严格复现“Adapt Pilot BCE + Reward 门控 + 投影 + 跨帧持久”；外循环优化未来 K 个 Data 窗口的 BCE/BER，并惩罚 Online 落后 Frozen。
- 可学习：`B_phi`、`W_phi`、`Q/R`、`a_0`、`Sigma_0`；不要只学 `a_0`。

**最小实验（两步）**

1. **只读子空间捕获诊断（不训练）**：对每个开发帧，计算 Data BCE 对 `delta_z` 的梯度 `g_data`；分别用
   - Adapt Pilot Fisher 的 top-r 特征向量（我们过去用的 prompt 子空间）；
   - Data 梯度 SVD 的 top-r 方向（query oracle 子空间）；
   - Bayes 后验条件化的 context 子空间（`B_phi(c)` 初始化）；
   
   测量三者对 oracle 单步增益的捕获率 `||P g_data||^2 / ||g_data||^2`，以及方向跨帧余弦稳定性。
   **判据**：若 prompt 子空间捕获率 <30% 且 query 子空间 >70%，说明我们的失败确实是 query 错配，值得训练；若 query 方向本身跨帧余弦 <0.3，停止。
2. **训练 + 配对评估**：4 臂（Frozen+Bayes、Online-Pilot-basis、Online-query-basis 固定、Online-context-conditioned），3 个新开发 seed × 64 帧 × 4 SNR；配对 `ΔBER`，报告 16 帧分箱斜率与末窗−早窗。

**预期失败模式**

- Oracle head 增益在留出帧消失（我们已观察到），说明 query 方向不可跨帧迁移；
- context 特征（后验方差/innovation）与真实可约误差相关性低；
- r=2/4 太小，捕获率不足；
- 即使训练集上捕获率高，独立 seed 上方向翻转（v20–v22 的复现失败）。

**关键文献**：[W26] query-aware 子空间；[B25] 不确定性校准；[MH25] context/hypernetwork 权重生成；[R24] LMMSE 初始化避免局部极小。

---

### 6.2 M2：软 Data 似然（ELBO）+ 可靠符号门控

**为什么可能仍有增量**

- Pilot 只有 224 个 Adapt 符号，且高 SNR 下 `p→b`、Fisher `p(1-p)→0`，梯度消失；Data 有 768 个符号，是唯一独立于 Pilot 的信息源。
- 软边缘化（对星座后验求期望）而不是硬伪标签，能避免“错误判决自我强化”；文献 [BB24][CAT26][SIR26] 都报告了少 Pilot/无标签 payload 的增益。
- 我们的 ELBO 诊断在 5/10/15 dB 的 `cos(ELBO, Data)` 约 `+0.15/+0.14/+0.24`，方向不是完全错误；问题是范数太小（15 dB 约 `9e-4`）且单步没有 BER 收益。若把它作为**低维 head 的辅助项**（而不是单独替代 Pilot），并配合可靠符号门控，可能把这点方向信号放大。

**参数化（草图）**

```python
# 单载波 BPSK，已知/估计 CIR h，冻结均衡器给出 mu_i, var_i
L_data = (1 / (2 * sigma2)) * (
    || y - H @ mu ||^2 + sum_l ||h_l||^2 * Var_{t-l}
)
# 可靠符号门控：只用后验熵低/重构残差小的 Data 符号
w_i = sigmoid((tau - entropy_i) / T) * reliability_i
L = BCE_adapt + gamma * sum_i w_i * L_data_i
# 梯度只回传到 r 维 a_t；h 固定或强锚定
```

- 可在 `L_data` 上加后验熵下限或一致性正则（同一 waveform 的不同 dropout/温度视图），但**熵最小化本身有塌缩风险**，必须与星座先验 KL 和 Reward 门控一起用；
- `gamma` 做单因素扫描（0/小/中），不要按 SNR 调度；
- `tau` 用离线开发集校准，不用 Data 标签；
- Reward Pilot 做接受/回滚；参数位移上限 `delta`；
- 离线外循环用未来 K 帧 Data BCE/BER，不用单帧。

**最小实验**

1. **梯度诊断（只读）**：逐帧计算 `g_elbo` 与 `g_data_oracle` 的余弦、范数、以及 `g_elbo` 在 Adapt Pilot Fisher 子空间外的分量；在 0/5/10/15 dB、3 seed × 16 帧。
   **判据**：5/10 dB 的 `cos > 0.3` 且 `||g_elbo||` 高于参数噪声底；否则停止（我们已有一次单步负结果）。
2. **固定半径单步 + 配对评估**：只更新 2 维 head，比较 `gamma=0` vs `gamma>0`；3 seed × 16 帧。
3. 若单步有同向信号，再做离线外循环训练与 64 帧评估。

**预期失败模式**

- Data 残差被不可约噪声/模型误差主导，ELBO 只是在拟合噪声；
- 软后验过度自信，边缘化退化为硬判决；
- `gamma` 稍大就压过 Pilot，导致有害更新；
- 我们的单步 ELBO 已经负，可能根本没有可用信号。

**关键文献**：[BB24] 半监督 VAE；[CAT26] 每 block 半监督 ELBO；[SIR26] 物理重建 + 伪导频；[ODG25] 数据辅助虚拟导频；[G26] 单步贝叶斯更新。

---

### 6.3 M3：学习式可靠性/符号方差模型（R_t 或 Q/R 的神经修正）

**为什么可能仍有增量**

- Bayes CIR 的 Data 后验噪声用 `R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}`，假设符号误差独立、高斯；0 dB 下误差相关、软判决偏差大，这个近似可能过于乐观。
- 0 dB 的 H3（增长）不通过：16 帧分箱先增长到约 `+0.037` 后回落，说明后验在后期可能“过度自信/错误传播”。这正是 Q/R 或符号方差模型失配的典型表现。
- 文献中 SKF 用 NN 动态调 Q/R（相对 LS 低 SNR 5 dB、高 SNR 3 dB）；Split-KalmanNet/SEKF 学习增益/过程噪声，强调对模型失配的鲁棒性。[SKF25][SK25][C]

**参数化（草图）**

```python
# Bayes CIR 的 Data 后验噪声
R_t = sigma2 + alpha_theta(f_t) * sum_l |h_l|^2 * Var_{t-l}
f_t = [abs(mu_t), Var_t, abs(innovation_t), local_pilot_bce, trace(Sigma_h)]
alpha in [0.5, 2.0]   # 有界，防止退化为 0 或无穷
# 可选：学习 Q 的 2 维修正（过程噪声），跨帧持久
```

- `alpha_theta` 是 2 层小 MLP（输入 5 维，输出 1 维），离线训练；
- 在线只前向（不改参数）或只更新 2 个标量；若要求“神经参数实际改变”，让 `alpha_t` 由 Adapt Pilot 的局部 BCE 做一次标量校准；
- Data 标签只在离线外循环/事后评分使用。

**最小实验**

- 0 dB、5 个新开发 seed × 128 帧；比较固定 `R_t`、学习 `alpha`、oracle `R_t`（用真实符号误差，只作上界）；
- 指标：BER、16 帧分箱斜率、后验协方差迹、错误符号的 `alpha` 分布；
- **判据**：0 dB 的 16 帧斜率转正且 BER 增益 > 0；5/10/15 dB 不退化；若 oracle `R_t` 相对固定 `R_t` 的 BER 差 < 噪声底，停止。

**预期失败模式**

- 仿真噪声是白高斯，符号误差近似独立，学习不到东西；
- `alpha` 过拟合开发 seed；
- Bayes CIR 已接近 oracle CIR，改进 `R_t` 的 BER 上限很小；
- 0 dB 的错误传播不是靠方差模型能修的（需要更好的符号先验/检测）。

**关键文献**：[SKF25] 学习 Q/R；[SK25] SNN 学习增益；[G26] 贝叶斯状态空间；[R24] 局部极小与初始化。

---

### 6.4 M4：物理参数化残差 impairment（相位/timing/tail/额外 tap）

**为什么可能仍有增量**

- HybridDeepRx 的增益来自**未建模的 PA 非线性**：在高 EVM 下明显优于线性接收机。[HDR21]
- 相位噪声专用接收机把 JCPD 展开，说明“物理模型 + 可学习参数”在存在确定性 impairment 时有效。[PN25]
- 我们的 P0 tail 对齐发现：旧路径的 Adapt Pilot BCE 在 10/15 dB 是 `0.0282/0.0357`，对齐后降到 `0.0067/0.0003`；Frozen BER 在 0/5 dB 略好。这说明**帧边界 tail 是一个真实的模型误差**，修掉后 Frozen 变好，但没有带来 Online 增量。
- 如果还有其他确定性残差（残余 CFO、timing、超过 116 tap 的截断、DAC/ADC 非线性），它们是可辨识、跨帧持续的偏差；学习一个小修正比学习噪声更容易。

**参数化（草图）**

```python
# 方案 A：逐帧相位/定时残差
y_corr[n] = exp(1j * theta_t) * (y[n] - delta_t)
# theta_t, delta_t 由 Adapt Pilot 的相位线/相关峰估计，或用 2 维小网络从 Pilot 特征预测

# 方案 B：额外 tail tap
h_extra[0:L]  # 在 Bayes CIR 之外增加 L 个可学习 tap，强锚定到 0

# 方案 C：符号级残差修正
delta_z[n] = small_mlp(features[n])  # 有界，零初始化
```

- 所有参数零初始化/强锚定，确保 Frozen 等价；
- 在线只用 Adapt Pilot 更新 2–4 维；Reward Pilot 门控；
- 离线用 Data 标签训练预测器（Data 只在外层）。

**最小实验**

1. **oracle 残差诊断（只读）**：逐帧用真实 Data 符号拟合 `theta_t`、`delta_t`、额外 tail tap，测 BER 上界；如果 oracle 增益在任一 SNR > 噪声底，说明存在可修残差；否则停止。
2. 若有 oracle 增益，用 Adapt Pilot 特征训练 2 维预测器，比较 Frozen/预测器/oracle；3 seed × 64 帧。

**预期失败模式**

- 仿真信道是理想线性 FIR + AWGN，没有 PA/PN/timing 残差；
- tail 已在 P0 修好，剩余 oracle 增益为零；
- 相位被 Bayes CIR 的逐帧 CIR 吸收，无法单独辨识；
- 学到的修正过拟合 Pilot 噪声。

**关键文献**：[HDR21] PA 非线性；[PN25] 相位噪声展开；[T26] 基线陷阱（CFO 补偿可能制造虚假增益）；P0 tail 对齐结果。

---

### 6.5 M5：Bayes 后验条件化 hypernetwork（无在线梯度）

**为什么可能仍有增量**

- 高 SNR 下梯度消失、小参数网络局部极小；hypernetwork 用一次前向生成权重，绕开在线梯度和局部极小。[MH25]
- 它可以把 Bayes CIR 的后验均值/协方差、innovation、SNR 作为 context，生成 2–4 维低秩修正；权重逐帧变化，满足“在线神经参数实际改变”的形式要求。
- 文献报告 modular hypernetwork 在时变信道下优于 joint training，并**接近 online training 的 SER**，复杂度显著更低。[MH25]

**参数化（草图）**

```python
# 离线训练；在线只前向
context = [h_bayes, diag(Sigma_h), innovation, snr_hat, pilot_features]
B_t, a_t = HyperNet_phi(context)      # 低秩基 + 系数，或直接 FiLM gamma/beta
delta_z = B_t @ a_t
```

- 输出层零初始化，保证初始 Frozen 等价；
- 离线多任务训练：每个信道 realization 是一个 task；外循环用未来 K 帧 Data BCE/BER；
- 在线不做梯度，只前向；如果要求“更新”，可以让 `a_t` 由 Adapt Pilot 做一次标量校准。

**最小实验**

- 训练 5 个开发 session，评估 2 个新 session；比较 Frozen+Bayes、EKF adapter、hypernetwork；
- 指标：BER、延迟、参数数；
- **判据**：至少一个 SNR 的配对区间不跨零；若只能匹配 EKF adapter 且延迟更低，可作为工程替代，但不能作为“神经增量”证据。

**预期失败模式**

- Hypernetwork 不能超过 teacher（online training）；如果 teacher 的增益 ~0，hypernetwork 也 ~0；
- 离线覆盖不足，OOD 失效；
- 输出维度/容量太小；
- 训练不稳定。

**关键文献**：[MH25] modular hypernetworks；[W26] query-aware 子空间；[B25] 不确定性条件化。

---

### 6.6 M6：学习式优化器/预条件

**为什么可能仍有增量**

- 我们的 GN/近端/Adam 都用了固定步长/固定预条件；不同 SNR/帧龄的最优步长可能不同。
- 文献中 learned optimizer 在无线 beamforming 中学习投影梯度下降的超参，保持可解释性并减少迭代。[C]
- 如果更新方向本身有信号，学习步长/预条件可能把微弱信号放大；如果方向没有信号，它也无能为力。

**参数化（草图）**

```python
# 离线 meta-train phi；在线只前向
s_t = [grad_norm, fisher_eigs, reward_bce, snr_hat, frame_age]
a_t = a_{t-1} - eta_phi(s_t) * P_phi(s_t) @ g_t
```

**最小实验**

- 离线 replay 现有 v24–v27 轨迹，比较“固定 GN”与“学习预条件”的每步 Data-BCE 下降；
- **判据**：每步下降 > 固定 GN 20% 且跨 seed 稳定；否则停止；
- 若有信号，再在线 3 seed × 64 帧。

**预期失败模式**

- 没有信号可学；学到的预条件只是过拟合开发 seed；
- v24–v27 已经显示 GN/近端无稳定收益；
- 计算/显存开销。

**关键文献**：[R24] 初始化比更新规则更关键；[G26] 自然梯度；learned optimizer for beamforming [C]。

---

### 6.7 M7：贝叶斯集成/不确定性校准（低优先级）

**为什么可能仍有增量**

- 模块化贝叶斯在 scarce data 下改善校准与 SER（QPSK 0.2 dB、8-PSK 1 dB）。[B25]
- 但关键细节：`Q=1` 时 SER 不变，BER 改善来自解码器；我们无编码，单调校准不改变硬判决符号。

**最小实验**

- 测 ECE/Brier 与 BER；若 BER 不变，只保留为诊断/门控特征，不作为增益机制；
- 若要与 C1 结合，把不确定性作为 context，而不是输出校准。

**预期失败模式**

- 无解码器，校准无出口；
- 集成计算成本高；
- 先验/似然失配时贝叶斯校准本身不可靠。

**关键文献**：[B25]；FiLM-Ensemble [C]；[LLR24] 需要 LDPC。

---

## 7. 与我们已失败实验的对接

| 已有结果 | 文献解释 | 下一步是否值得 |
|---|---|---|
| v12/v13 Pilot-only 风险门控失败 | [W26] prompt fit ≠ query；[B25] 校准需解码器 | 不再单独调门控；门控只作安全层 |
| v14 配对 Online 增益外层目标失败 | 外层目标不能创造内层信号；Bayes gap 未变 | 不再只改外层损失 |
| v15 Frozen 错判条件化失败 | 离线分桶不改变在线硬判决；小参数局部极小 | 不再用分桶权重 |
| v16/v17 12 维子空间 + 梯度传输失败 | [W26] 固定子空间 mismatch floor；[R24] 局部极小 | 先做 query-alignment 诊断，再决定子空间 |
| v20/v21/v22 Pilot 上下文 adapter 失败 | prompt-aligned context；单帧历史不足 | 换成 Bayes 后验 context + 多帧外循环 |
| v24–v27 GN/近端元学习失败 | 梯度方向本身无信号；learned optimizer 不能创造信息 | 先做可辨识性审计；不扩 seed |
| 全 head Adam/trajectory 失败 | 高维适配过拟合；无持久 query 方向 | 固定低维 + query-aligned |
| 共享 Bayes CIR 后 head +1 bit | [W26] 剩余是不可约后验方差；[L26] BCRB | 这是负结果的核心证据；不要用“再换损失”绕过 |
| 单步 ELBO 无稳定 BER | ELBO 范数小、模型误差主导；[BB24][CAT26] 需大量 payload/解码 | 先做梯度诊断；只在 5/10 dB 尝试 |
| P0 tail 对齐改善 Frozen | 真实模型误差在帧边界；修掉后 Frozen 变好 | 检查是否还有残余 impairment；但不要期待 Online 增量 |

**建议的最小实验路径（按顺序，任何一步失败就停）**

1. **只读诊断 A（1 天）**：query-vs-prompt 子空间捕获率 + 跨帧方向稳定性；ELBO 梯度余弦/范数；oracle 残差 impairment 上界。三个诊断都不训练，只用现有 checkpoint 和开发 seed。
2. **只读诊断 B（1 天）**：0 dB 的 `R_t` oracle 上界（用真实符号误差）与固定 `R_t` 的 BER 差；若差 < 噪声底，M3 停止。
3. **M1 训练（若诊断 A 通过）**：2–4 维 context adapter，3 新 seed × 64 帧；4 臂对照；配对统计。
4. **M2 训练（若诊断 A 的 ELBO 通过）**：ELBO 辅助 + 可靠门控；先单步，再 64 帧。
5. **M4（若诊断 A 的 oracle 残差通过）**：物理残差预测器。
6. **M5/M6/M7 只在 M1/M2 出现同向信号后**作为增强；否则不做。

**预注册建议**

- 主 SNR：0/5/10/15 dB 全部报告，不根据开发结果删点；
- 主效应：配对 `BER_online - BER_frozen`；每点最大退化 0 作为护栏；
- 增长：固定窗口斜率 > 0、末窗优于早窗、会话级检验；允许局部窗口回升，不要求逐帧单调；
- 样本量：至少 3 seed × 64 帧用于机制筛查；正式确认按既有功效分析（0 dB 30–35、5 dB 10–15、10 dB 20–25、15 dB 10–15 session × 128 帧）；
- 统计：配对 bootstrap/permutation；多重比较校正；报告效应量和 CI；
- 数据角色：Adapt Pilot 更新；Reward Pilot 验收；Data 标签只用于离线训练/评分；不进入在线门控；
- 计算：报告在线参数数、更新次数、接受/回滚次数、每帧延迟/显存；Frozen 不必执行无用梯度。

---

## 8. 公开代码/数据集/benchmark

| 资源 | 内容 | 可用性 | 核验 |
|---|---|---|---|
| [gusakovy/adaptive-deep-receivers](https://github.com/gusakovy/adaptive-deep-receivers) | 在线贝叶斯深接收机（CM-EKF/Lo-Fi/Vd-EKF）实现，含 streaming fit 接口 | 公开 GitHub | A（API 核验存在） |
| [Rohde-Schwarz/NeuralReceiver](https://github.com/Rohde-Schwarz/NeuralReceiver) | DeepRx 风格神经接收机 + LS/ZF 基线，基于 Sionna PHY；可配置 TDL、SCS、MCS、LDPC | 公开 GitHub | A（API 核验存在） |
| [NVlabs/sionna](https://github.com/NVlabs/sionna) | 链路级仿真库；官方 Neural Receiver tutorial 是事实参考实现 | 公开 GitHub | A（API 核验存在） |
| [Ewwftyy/neural-receiver-5g-nr](https://github.com/Ewwftyy/neural-receiver-5g-nr) | “Training Coverage” 论文的代码/配置/结果；含 genie/practical LMMSE 与 hybrid 残差修正 | 公开 GitHub | A（API 核验存在） |
| [CENTRIC Dataset #2 (Zenodo 15856694)](https://zenodo.org/records/15856694) | 5G 上行硬件采集（1×4 SIMO、2 UE、132 PRB、16-QAM、3.7 GHz），含接收 slot 与真值比特 | 公开数据集 | A（Zenodo API 核验） |
| [tnd-lab/Neural-Network-based-Information-Theoretic-Transceivers](https://github.com/tnd-lab/Neural-Network-based-Information-Theoretic-Transceivers) | NN 比特级解映射/LLR 接收机与 BER 分析 | 公开 GitHub | A（API 核验存在） |
| Sionna Neural Receiver tutorial | 端到端 DeepRx 示例、BER/BLER 评估 | 公开文档 | B（未本次打开） |
| 3GPP TR 38.901 TDL/CDL | 标准信道模型 | 公开标准 | B（文献引用） |

> 注意：这些资源大多是 OFDM/MIMO 设置，不能直接复现我们的单载波 116-tap 长回波；可复用的是**评估协议、genie/practical 基线思想、LLR 残差修正结构、在线贝叶斯更新接口**。

---

## 9. 参考来源与核验状态

### A 级（本次全文打开并阅读）

1. T. Wakayama. *A Decision-Theoretic View of Test-Time Training: When, How Far, and Which Directions to Adapt.* ICML 2026, PMLR 306:124279–124317. arXiv:2606.15569. <https://arxiv.org/abs/2606.15569>
2. L. Lagona, C. Bockelmann, A. Dekorsy. *On the Bayesian Performance Limits of Pilot-to-Prediction in Frequency-Selective OFDM Systems.* IEEE Open Journal of the Communications Society, 2026. DOI: 10.1109/OJCOMS.2026.3721910. <https://www.ant.uni-bremen.de/sixcms/media.php/102/15381/On_the_Bayesian_Performance_Limits_of_Pilot-to-Prediction_in_Frequency-Selective_OFDM_Systems.pdf>
3. M. Obeed, M. Jian. *Learning During Detection: Continual Learning for Neural OFDM Receivers via DMRS.* arXiv:2602.20361 (2026). <https://arxiv.org/abs/2602.20361>
4. T. Raviv, N. Shlezinger. *Modular Hypernetworks for Scalable and Adaptive Deep MIMO Receivers.* IEEE, 2025. arXiv:2408.11920. <https://arxiv.org/abs/2408.11920>
5. T. Raviv, S. Park, O. Simeone, N. Shlezinger. *Uncertainty-Aware and Reliable Neural MIMO Receivers via Modular Bayesian Deep Learning.* IEEE Trans. Veh. Technol. 74(11):17637–17651, 2025. DOI: 10.1109/TVT.2025.3579989. arXiv:2302.02436. <https://arxiv.org/abs/2302.02436>
6. *The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models.* arXiv:2506.24000 (NeurIPS 2025 D&B). <https://arxiv.org/abs/2506.24000>
7. V. Rozenfeld, D. Raphaeli, O. Bialer. *Enhancing LMMSE Performance with Modest Complexity Increase via Neural Network Equalizers.* arXiv:2411.01517 (2024). <https://arxiv.org/abs/2411.01517>
8. S. Ha, I. Lee, S. Jeon, Y.-S. Jeon. *Online Data Generation for MIMO-OFDM Channel Denoising: Transfer Learning vs. Meta Learning.* arXiv:2508.09751 (2025). <https://arxiv.org/abs/2508.09751>

### B 级（本次仅摘要/记录页）

9. Q. Hu, F. Gao, H. Zhang, S. Jin, G. Y. Li. *Deep Learning for MIMO Channel Estimation: Interpretation, Performance, and Comparison.* arXiv:1911.01918 (2020). <https://arxiv.org/abs/1911.01918>
10. *Training Coverage, Not Architecture, Decides When Learned 5G NR Receivers Help.* Zenodo 22994791 (2026 preprint). <https://zenodo.org/records/22994791>
11. R. Luostari, D. Korpi, H. Holma, O. Tirkkonen. *Cross-Site Transfer and Spatial Forgetting in OTA-Trained Neural Receivers.* arXiv:2610.02577 (2026). <https://arxiv.org/abs/2610.02577>
12. M. Zanella et al. *Realistic Test-Time Adaptation of Vision-Language Models.* CVPR 2025. arXiv:2501.03729. <https://arxiv.org/abs/2501.03729>
13. J. Pihlajasalo et al. *HybridDeepRx: Deep Learning Receiver for High-EVM Signals.* IEEE PIMRC 2021. DOI: 10.1109/PIMRC50174.2021.9569393. arXiv:2106.16079. <https://arxiv.org/abs/2106.16079>
14. X. Xiao et al. *Spike-Kal: A Spiking Neuron Network Assisted Kalman Filter.* arXiv:2504.12703 (2025). <https://arxiv.org/abs/2504.12703>

### C 级（仅检索摘要/二手页面，未打开原文；不用于关键结论）

15. *Low-Complexity and Low-Energy RNN-Enhanced Kalman Filter for Channel Estimation: A Case Study.* IEEE, 2025, doc 11397385. <https://ieeexplore.ieee.org/document/11397385>
16. H. Shen, B. Li, Y. Sun, W. Xu, H. Zhang, C. Zhao. *Learnable Semi-Blind Receiver Design for Phase Noise Impaired OFDM Systems.* IEEE Trans. Commun. 73(12):13846–13862, 2025. <https://ieeexplore.ieee.org/abstract/document/11187365>
17. *Deep Learning Aided LLR Correction Improves the Performance of Iterative MIMO Receivers.* IEEE Trans. Veh. Technol. 73(2):2045–2060, 2024. DOI: 10.1109/TVT.2023.3312029. <https://ieeexplore.ieee.org/document/10246845>
18. *FedPDA: Collaborative Learning for Reducing Online-Adaptation Frequency of Neural Receivers.* IEEE INFOCOM 2025. <https://ieeexplore.ieee.org/document/11044747>
19. G. Revach et al. *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics.* IEEE Trans. Signal Process. 70:1532–1547, 2022. <https://ieeexplore.ieee.org/document/9715175>
20. Y. Liu et al. *TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?* NeurIPS 2021. <https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html>
21. Z. Wang, X. Chen, G. Li, H. Ji, T. Zhang. *When Test-Time Training Fails: A Critical Analysis of Robustness and Hyperparameter Sensitivity.* TMLR 2026. <https://experts.illinois.edu/en/publications/when-test-time-training-fails-a-critical-analysis-of-robustness-a/>
22. *SICNet / DeepIM-SIC*（完美 CSI 下经典 SIC 最优、神经仅接近；增益在不完美 CSI）——检索摘要，未逐篇核验。

### D 级（前次 v1 已全文核验，本次未重开）

23. Y. Gusakov, O. Simeone, T. Routtenberg, N. Shlezinger. *Online Learning of Modular Bayesian Deep Receivers.* IEEE Trans. Signal Process. 74:2192–2207, 2026. arXiv:2511.06045. <https://arxiv.org/abs/2511.06045>
24. T. Raviv, S. Park, O. Simeone, Y. C. Eldar, N. Shlezinger. *Online Meta-Learning for Hybrid Model-Based Deep Receivers.* IEEE Trans. Wireless Commun. 22(10):6415–6431, 2023. DOI: 10.1109/TWC.2023.3241841. arXiv:2203.14359. <https://arxiv.org/abs/2203.14359>
25. N. Burshtein, E. Bery. *Semi-Supervised Channel Equalization Using Variational Autoencoders.* IEEE Trans. Wireless Commun. 23(12):19681–19695, 2024. DOI: 10.1109/TWC.2024.3485991. arXiv:2309.11841. <https://arxiv.org/abs/2309.11841>
26. A. Caciularu. *Geometric Inductive Biases for Semi-Supervised Equalization: The Constellation-Aware Transformer.* arXiv:2609.33695 (2026 preprint). <https://arxiv.org/abs/2609.33695>
27. Y. Shi, X. Qi, Y. Chen, et al. *SIRIUS: Unsupervised Online Channel Estimation for High-Mobility OFDM via Implicit Neural Representation.* arXiv:2605.10213 (2026 preprint). <https://arxiv.org/abs/2605.10213>

---

## 10. 给主 agent 的三句话

1. **先承认理论边界**：Bayes CIR 已把 CIR 相关 Bayes gap 吃掉，剩余是不可约后验方差；BCRB 和 BCJR 最优性都说明“同资源、同分布、无编码”下神经在线增量的先验接近零。这不是失败，而是需要如实报告的核心负结果。
2. **如果还要再试一次，只做 M1（query-aligned 不确定性条件化 2–4 维残差）**：先用**只读诊断**测 query-vs-prompt 子空间捕获率与跨帧方向稳定性；捕获率不过 70% 或方向不稳定就停，不训练、不扩 seed。M2（软 Data ELBO）作为第二候选，但必须先用梯度余弦/范数过门槛。
3. **不要再做**：换外层损失、换门控阈值、扩同结构 seed 矩阵、无编码下的校准/集成、没有 oracle 证据的物理残差修正。若 M1/M2 都失败，正确结论是“物理 Bayes CIR 在线增益成立、神经参数增量未成立”，并按 AGENTS 契约保留该负结果，不用新结构掩盖。

---

> 本报告不构成“核心目标已达成”的证据。所有文献增益均为外部设置；我们的核心验收仍是同源、同物理状态跟踪 Frozen 的配对 BER 增益，以及随帧数增长的可重复趋势。

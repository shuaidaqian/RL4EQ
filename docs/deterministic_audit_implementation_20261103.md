# 确定性神经微调与同波形开发审计实现

本阶段落实[路线 v3](research_route_v3_20261103.md)的首个长回波归因锚点。新入口不创建策略、离散动作或 PPO 状态；已有 RL 入口保持复现用途。本文件描述实现，不把开发数据当作独立确认。

## 固定开发协议

清单为 [adaptability_long_anchor_v1.json](../configs/audits/adaptability_long_anchor_v1.json)：沿用 `eme_long_memory_v2` 和既有普通监督 control64 checkpoint，保持 1024 符号、116 记忆与接收后缀、Adapt224/验证32。使用训练分布 `offline_train` 的新环境 seed 120300–120302，完整覆盖 0/5/10/15 dB。从会话开始只做一次 acquisition，每条连续 128 帧，共 1536 条主评分记录。

这是同时含 CIR 漂移和同步残差的既有长回波锚点，不是单因素机制实验。短/中记忆的独立机制配置及离线新结构训练仍属后续任务，不以此锚点代替。

烟测使用相同开发 seed 的前两帧，仅检查实现。完整运行重用这些开发前缀，不能声称整批都是未观察的独立确认数据。停止规则按固定预算执行，协议、指纹或非有限数失败才中止；不按性能追加 seed。

## 接收分支与归因

`evaluation/adaptability_audit.py` 在控制开始前将普通帧的 Data 发送符号清零、bit 清零、真实 CIR/tail 移除，只保留合法 Pilot 与已接收波形。可部署控制不会收到逐样本真实相位。

|分支|当前状态与更新|解释|
|---|---|---|
|F|acquisition CIR 固定，仍有当前 Pilot 条件适应、相位校正和冻结神经支路|固定 CIR 的正常条件基准，不等于完全不适应|
|P|合法 Adapt 方程、全协方差递推与相同冻结网络|状态跟踪改变的组合效果|
|N|与 P 完全相同的物理上下文，加确定性神经记忆更新|N 对 P 隔离神经参数更新增量|
|T|同权重网络，隔离替换实际同步相位与当前 CIR|诊断状态失配，不是最优性能界|

所有分支共用由 P 整帧接收判决得到的 soft tail，首帧 tail 为已知 acquisition 尾部。N 不改变 P 的跟踪和尾部，因此 N 对 P 不混入后续物理状态变化；这也意味着它是受控归因实验，还不是最终自主接收机架构。F 的 tail 同样由 P 提供，不能冒称完全复现旧 `Frozen Offline NN` 的独立会话实现。

P 每帧消费一次新的 Adapt 似然，按声明的 `rho_frame` 预测均值/协方差。过程功率从 acquisition 估计幅度和噪声构造，不读取真实 PDP；acquisition 协方差是对角近似，不能称为已校准真实后验。相位坐标复用冻结网络的校正，P 与 N 一致。网络既有的冻结 Pilot 条件状态支路仍正常执行。

## 确定性神经更新

`training/deterministic_adaptation.py` 仅训练 `channel_state.memory.weight` 的 234 个实数元素，其余神经权重冻结。它沿用已有稠密记忆作为首个诊断候选，不宣称已选定最终低维结构。

- Adapt 目标为完全已知卷积行上的归一化信号误差，加相对本帧起点的近端项；验证 Pilot 不进入梯度。
- 每帧重置 AdamW，学习率 .02，3 步，梯度范数上限 1，无 weight decay。
- 每步先投影到本帧起点半径 .15，再投影到会话初始参数半径 .8。
- 候选完成后，验证 Pilot 信号损失至少改善 `1e-5`，且验证硬 BER 不升，才持久保留；否则恢复本帧起点参数。
- 完整记录守门前候选和守门后输出，包括拒绝帧。没有以 Data 结果控制更新或选择候选，也没有未来帧回滚策略。

本轮未重新训练离线网络。因此，即便发现 N 增益，也只能支持此 checkpoint 与更新组合的开发机制，不能直接证明新离线目标或元学习优于普通监督。

## 一次候选有效期与真状态限制

首帧保存原始候选及守门后候选两份只读网络，此后不追加神经更新。在偏移 0/1/8/32/128 帧用与 P 相同的合法物理状态评分。偏移 128 额外接收一帧，仅写入 `fixed_candidate_metrics.jsonl`，不混入 128 帧主平均、不增加主分支神经更新。未来 Pilot/Data 评分均不返回在线控制。

仿真器增加只读 `last_phase_used()`；原相位函数默认返回值和随机流保持兼容。诊断环境在实际发送时缓存 CIR/相位，避免 lookahead 推进后把下一帧真状态错给当前帧。普通 `Frame` 与 `receiver_view` 没有新增真相位字段。

T 仍不知道 Data 与真实前帧 tail。接收算子对后缀仍使用当前 CIR 的近似；另记录下一帧真实 CIR 与当前 CIR 的差值，仅供事后诊断。T 的失败不能当作不可解性证明。

## 记录与验证

主日志包含波形/后缀哈希、物理帧号、四分支与原始候选 BER、验证指标、梯度与参数变化、拒绝、后验方差、单帧及历史信息矩阵秩/条件数。局部神经 Pilot Jacobian 在帧 0/7/127 单独记录，欠定时完整参数空间条件数记为空，不把非零奇异值条件数冒称全空间可辨识。

计时使用 GPU 同步：F/P 分别记录各自当前条件和检测，N 记录 P 流程加额外候选/梯度/守门，T 是诊断前向。计时不含环境、acquisition、评分与信息几何。由于存在共享 P tail，这些是当前受控实现的分支成本，不能直接当作各方法独立部署的端到端延迟。

运行时保存清单、配置/checkpoint 指纹及相关源码指纹；运行结束再次检查源码未改变，拒绝覆盖输出。正式应用范围和统计门槛继续待冻结。

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -m pytest -q -p no:cacheprovider
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method audit --audit-manifest configs/audits/adaptability_long_anchor_v1.json --audit-smoke --output-dir logs/adaptability_long_anchor_v1_smoke_v2_20261103
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method audit --audit-manifest configs/audits/adaptability_long_anchor_v1.json --output-dir logs/adaptability_long_anchor_v1_20261103
```

实现阶段全量回归为 **878 passed，372.87 秒**；其后新增 CLI 防误用检查与相关确定性审计、文档契约的定向回归为 **17 passed，13.35 秒**。首次全量检查发现旧文档测试仍强制已撤销的 RL 联合门槛，更新该契约后重跑全量通过；未通过修改历史科研结果来消除失败。

最终版四 SNR、3 seed、每条2帧的 GPU 烟测完成：24条持续分支记录及24条一次候选诊断记录，偏移覆盖0/1帧。接收、参数更新、守门、真状态快照与输出指纹检查通过，[烟测元数据](results/adaptability_long_anchor_v1_smoke_20261103.json)记录环境与源码指纹；不据此作性能收益判断。

完整开发矩阵结果与下一步判断在实验完成后另行归档。核心科研目标没有因接口实现而自动完成。

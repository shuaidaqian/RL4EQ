# 固定长回波的检测子空间：实现与开发预算

本阶段直接承接[固定信道结构计划](low_dimensional_mechanism_plan_20261103.md)，不新增或修改信道profile。实现完成不等于科研命题成立；运行结果另行归档。

## 结构与归因

来源为既有普通监督control64接收机。全网权重、CIR记忆、同步补偿与合法P跟踪均冻结为同一来源。通过一次前向的局部钩子读取已有符号残差支路的7维接收特征：复数线性估计、soft符号、复数匹配残差、噪声尺度、观测覆盖。钩子在异常时也删除，不改变来源网络结构、状态字典或前向结果。

新神经基为`7→16→k`的两层tanh网络，`k∈{2,4,8}`。输出为`来源logit + (.5/noise)/sqrt(k) × basis(features) × coefficients`。零系数精确保留来源；离线学习基网络与系数初始化，在线仅更新k个实数系数。该残差直接修正检测logit，不修改CIR、同步或物理跟踪状态；它不是把解析tap估计换一种存储形式。

`PilotObservation`在线接口不含Data bit、Data发送符号、真实CIR/同步或真实tail。Adapt用于BCE更新；验证Pilot仅在动作后作BCE严格改善且硬BER不升的守门。每帧重置Adam状态，3步、学习率.02、梯度裁剪1、局部半径.15、会话初始中心累计半径.8、近端.1、验证改善阈值1e-5。内循环函数在线与离线共用，硬守门保留实际分支，外层不对接受布尔值求导。Adam二阶矩平方根前仅增加1e-24下界，避免零梯度处高阶导数非有限；与实际AdamW数值一致性由测试检查。

P与各神经分支共享来源P输出产生的soft tail。这样有利于在线梯度归因，但尚非自主闭环部署验证。Frozen仍正常进行同样的逐帧Pilot物理与神经条件适应，不能将其称为无适应接收机。

## 门槛、数据角色与固定预算

[开发清单](../configs/audits/detection_subspace_long_v1.json)引用现有信道JSON与checkpoint的SHA256，不复制并改写信道字段；四SNR完整覆盖。

- 检查：seed121000/121001，每点8帧，共64帧。在帧0/7检查全部三个维数。要求全部指定条件满列秩、梯度有限、Data输出Jacobian非零、条件数不超过100，选择最大的合格维数，不按Data增益选维数。
- 同时复跑既有稠密记忆的64帧短诊断，补齐更新前、候选、守门后、来源重置反事实的只读Data评分。重置反事实等于P，因为分支只改了memory.weight；测试核验实际重置与P输出相同。该反事实不重置持续运行分支，也不能定位某次历史更新的全部因果影响。
- 普通监督作用检查：选定结构的独立副本，检查seed121000的首帧、完整四SNR，固定8次离线更新。只证明系数、基与Data输出能够变化，不证明在线泛化；不把副本当四臂训练初始化。
- 训练：seed121100–121103，每SNR9帧，前8帧更新，第9帧只提供未来离线评分，共144帧。
- checkpoint选择：seed121200/121201，同样每SNR9帧，共72帧。第16/32/48/64次外层更新使用相同选择标准：Frozen与Online完整四SNR等权Data BCE平均，严格改善才替换，平局取早期。
- 开发评估：模型选定后生成seed121300–121302，每SNR32帧，共384帧、A/B共768条配对方法记录。它们仍是开发数据，不是正式独立确认，也不能替代128帧或数百秒持续会话。

A普通监督与B更新后训练均从同一种子、同一零系数结构开始，均64次外层AdamW更新、学习率.001、weight decay为0，按相同顺序消费相同训练序列与Data标签。A评分每个时刻及下一帧；B按实际Adapt/验证/持久流程更新后评分同样的当前与下一帧。离线只学习残差基与系数初始化，来源大网络冻结。外层更新次数和标签匹配，不声称两者计算量相等，实际耗时另报。

对选定A/B checkpoint分别运行Frozen/Online，报告在线增量、离线基础变化及交互，完整保留拒绝、有害接受和旧记忆相对Frozen退化的帧。统计单位为独立信道seed，3-seed配对t区间仅为探索性描述，未经多重比较校正。

## 入口与停止

```powershell
python online_train.py --method subspace_audit --audit-manifest configs/audits/detection_subspace_long_v1.json --output-dir logs/detection_subspace_inspection_v1_20261103 --device cuda
python pretrain.py --stage detection_subspace --subspace-manifest configs/audits/detection_subspace_long_v1.json --inspection-dir logs/detection_subspace_inspection_v1_20261103 --save-dir logs/detection_subspace_training_v1_20261103 --device cuda
```

输出目录不得覆盖；不混用旧切片/训练覆盖参数。无合格维数或普通监督作用未通过时不训练；非有限数、角色隔离、来源指纹或协议错误停止并保留中间工件，不按正负结果增补seed。运行记录源码与配置/模型/缓存/轨迹指纹。

本实现复用冻结接收机特征，因此可精确缓存前向以加速系数训练；来源物理跟踪/特征提取与子空间更新分别计时。它们不是独立部署端到端延迟，接收等待、acquisition、仿真和评分开销单列边界。

## R2 第二轮：状态年龄安全分支

第一轮显示历史系数失效与验证Pilot误判同时存在。第二轮清单`configs/audits/detection_subspace_long_v2_reset_v1.json`只增加一个更新规则，信道、Pilot、seed角色、帧数和模型维数候选保持不变：动作完成后，用验证Pilot比较当前系数、候选系数和离线初始系数。若离线初始系数相对当前保留值的验证损失严格低于`1e-5`且验证硬BER不升，则将参数回退到该初始系数；Data标签不参与候选选择、回退或阈值。`candidate_accepted`与`reset_to_origin`分开记录，避免把安全回退误写成神经更新接受。

这个回退是在线可执行的状态年龄假设：在线系统只保留一个离线锚点和当前系数，验证Pilot发现当前记忆退化时回到锚点。离线B内循环和在线更新使用同一个分支；A仍使用普通监督训练、同一四臂预算。第二轮在固定预算的开发评估中回退实际触发，A/B分别33/36次；但未形成可归因、可重复的Data增益。0 dB的B在线增益均值约`+5.43e-5`、95%区间跨零，B相对A交互为负；5/10 dB在线均值退化，15 dB无有效错误。详情见[第二轮结果归档](results/detection_subspace_long_v2_reset_v1_20261103/README.md)。

对第二轮768条评估帧做只读复算后发现，BER符号一致率受大量并列影响，不能用于判定校准。改用连续Data BCE后，接受候选的验证BCE改善与当帧Data BCE改善方向在0 dB约86%、5 dB约65%一致；但持久运行的Data BER仍没有收益，提示单步目标部分有效而跨帧累计可能抵消作用。Data BCE只用于动作后的离线诊断，不参与在线更新或控制阈值。

R2第三轮清单`configs/audits/detection_subspace_long_v3_decay_v1.json`只加固定0.5锚点收缩：验证守门先通过候选后，将候选向离线初始系数收缩一半，再用同一验证Pilot守门；B内循环与在线流程复用该规则。固定信道、Pilot、更新步数、A/B预算与seed不变。沿用已观察的开发seed只作单因素机制比较，不构成独立迁移证据；对应代码和测试已提交，GPU结果待运行。不能据此进入R3或宣称核心目标已完成。

## 验证状态

第一轮完整pytest为903 passed。第二轮实现后定向回归79 passed；GPU几何审计选择4维（2/4合格、8淘汰），四臂训练和768条A/B开发评估均已运行并归档。第三轮0.5锚点收缩实现后定向回归64 passed，GPU实验待运行。第二轮未支持核心在线Data增益，R2仍进行中；R3应用与统计门槛仍待冻结。

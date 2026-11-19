# 持续均衡 v2 执行记录

目标依据：[用户确认的完整研究路线](研究点二_第一性原理审视与研究路线_20261119.md)。本文件只记录新分支事实与下一步，不替换历史物理快照的证据。

## 已完成的仓库操作

- `bayes-cir`：当前修正与研究路线已提交为 `9516241` 并推送到 `RL4EQ/bayes-cir`。
- 提交前全量测试：`python -m pytest -q -p no:cacheprovider`，160 passed。
- 已创建并推送 `continual-equalization-v2`，跟踪同名远端。
- 实验环境使用 `.venv-gpu/Scripts/python.exe`：PyTorch 2.11.0+cu128，GTX 1650。系统默认 Python 是 CPU 版本。

## 需求与进度

| 阶段 | 需求 | 当前状态 |
|---|---|---|
| P0 | 第一研究点到离散信道的物理映射、状态尺度和信息契约 | 已完成来源/抽象审计，精确 PDP 和时间统计映射仍待补足 |
| P1 | acquisition 合法估噪、A/B/C 匹配对照、早期计分 | 已实现并通过定向测试，四臂 3×32 开发实验运行中 |
| P2 | Q/持久化/soft Data 贡献与可靠性 | 待验证 |
| P3 | 闭环离线重训、多训练重复、神经贡献与协同 | 未开始 |
| P4 | 物理边界与长会话、功效与饱和容差冻结 | 未开始 |
| P5 | 四 SNR 自身改善、优势增长、饱和与保持的新独立确认 | 未开始 |
| P6 | 成本报告与实现优化；明确预算后的实时验收 | 未开始 |

## 实验规则

- 所有新实验先在 `docs/protocols/continual_v2_seed_registry.json` 登记，检查已有结果 seed 和保守历史保留范围。
- 同一实验内同波形 A/B/C/D 比较属于预先登记的配对设计。不得把已经运行的开发 seed 变成独立确认。
- 输出记录开发身份、配置/源码/checkpoint 指纹、噪声来源、时序和前向次数；中断保留部分结果，不覆盖重跑。
- 新接收模块只接受已剥离标签、真实状态和未来帧的接收视图。评分在接收状态更新之外完成。
- 新方案先使用未归一化有效 CIR，避免直接投影后协方差的解释问题；旧归一化方法保持为历史锚点。初始相位由 acquisition CIR 吸收，当前公共相位由 Adapt 估计。

## 最近决策

先用既有 checkpoint 检验合法估噪与递归物理状态，不立即扩网络。当前 Pilot 先更新再单次解码；Data 后验只作用于后续帧，禁止重复计入同一批 Pilot 信息。尚未证实新研究目标。

## 当前实现与验证

- [物理映射审计](continual_v2_physical_mapping_20261119.md)区分第一研究点、雷达包络与当前构造性假设；核查生成器创新与历史均匀 Q 的量级差异，尚不据此更换超参数。
- `training/continual_receiver.py`：严格接收视图、acquisition CFO 粗网格＋局部细化、未归一化 LS 残差估噪、有效复 CIR、A/B/C/D 前缀 Pilot 时序及一次前向。
- `evaluation/seed_registry.py`：扫描历史结果并登记实验，拒绝种子冲突和隐式重跑。
- `evaluation/continual_summary.py`：以会话为单位分别汇总自身改善、配对增量和时间变化；拒绝未完成/缺帧/波形不匹配的完整结论。
- 新增噪声尺度、线性高斯递归一致性、隐藏标签隔离、种子重复和相对优势误归因测试。
- 本阶段全量测试：`python -m pytest -q -p no:cacheprovider`，**170 passed**。不以单元测试替代研究收益验证。

## 运行中的开发实验

实验名 `p1_four_arms_estimated_noise_3x32`，seed 210001–210003，32 帧，四 SNR、四接收臂，全部是开发数据。精确设定噪声只保存为事后比值诊断；模型条件和所有后验使用 acquisition 估计。

```powershell
$env:PYTHONUTF8=1
& ./.venv-gpu/Scripts/python.exe scripts/develop_continual_receiver.py --checkpoint pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 210001 210002 210003 --frames 32 --experiment p1_four_arms_estimated_noise_3x32 --output docs/results/continual_v2_p1_20261119/four_arms_3x32.json --device cuda
```

**上述为已登记且已启动的命令，不能再次执行。** 继续工作时先检查现有进程与结果状态，运行完成后执行只读汇总：

```powershell
python scripts/summarize_continual_receiver.py --input docs/results/continual_v2_p1_20261119/four_arms_3x32.json
```

最近一次可核实运行句柄：工具 exec session `51194`。结果每 8 帧原子保存，文件 `status=running` 本身不足以证明进程仍活跃，须检查句柄或实际进程。完成后记录完整结果并更新本节。

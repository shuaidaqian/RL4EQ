# 固定rank-1神经状态滤波机制筛查

## 目的与协议

在不改 Level B `eme_long_memory_v2`、116符号长回波、Pilot=256（Adapt=224、Reward=32）及0/5/10/15 dB设置的条件下，比较每帧阻尼rank-1 GN与一个标量高斯神经系数状态滤波。两者共享强离线checkpoint、Pilot Fisher方向、Adapt BCE score、Reward硬门控和累计参数半径0.05。状态臂只增加均值系数与方差：用当前Adapt Pilot梯度/Fisher递推均值和协方差；Reward决定是否提交均值，拒绝时系数回滚但协方差继续吸收Adapt信息。

状态超参数预先固定为 `P0=100`、`Q=0.01`、信息缩放1、阻尼0.01、方差下限`1e-8`，四个SNR共享。该参数化尚未经过离线轨迹学习，结果仅用于机制与实现筛查。

## 实现核验

- 纯数值单测覆盖高斯状态更新的有限值、方差正下限、Adapt信息导致不确定度收缩、零score不移动均值，以及系数步长限幅。
- 端到端 smoke 见 `rank1_state_filter_smoke_v1_20261106.json` 与 `rank1_state_filter_smoke24_v1_20261106.json`。四SNR均完成；低SNR Reward接受首个候选，高SNR拒绝候选时系数保持不变，而方差按Adapt Fisher更新。24输出分块 Jacobian在4 GiB GPU上运行通过。
- 同session 4帧 A/B 见 `rank1_state_filter_dev4_v1_20261106.json`。0/5 dB两种更新器的配对BER收益相同，10/15 dB都无变化；状态滤波没有显示相对无记忆GN的增量。0/5 dB的Reward只接受首帧，接受时系数已触及累计半径0.05，之后拒绝使均值不再变化；方差则继续更新。
- 8帧评估曾在完成首个SNR前手动中止以控制GPU运行时，没有产生结果文件，不作为实验数据。

## 共同小步长筛查

为避免首帧候选耗尽累计半径，A/GN与B/状态滤波使用完全相同的单步上限0.005，在新session `76205` 上评估8帧和全部四SNR。两臂均使用中心有限差分估计同一方向导数（epsilon=0.001），每个参数提议由Adapt BCE/Fisher计算、Reward Pilot硬门控；该有限差分只为降低4 GiB GPU上的反向Jacobian成本，数值与解析线性单测对照通过。

| SNR | GN Online−Frozen | 状态滤波Online−Frozen | 两臂接受数 GN/滤波 | 有害接受 GN/滤波 |
|---:|---:|---:|---:|---:|
| 0 dB | +1.63e-4 | +1.63e-4 | 2/2 | 0/0 |
| 5 dB | 0 | 0 | 2/2 | 0/0 |
| 10 dB | 0 | 0 | 1/1 | 0/0 |
| 15 dB | 0 | 0 | 3/3 | 0/0 |

每SNR只有一个session、8帧，两臂全部硬BER逐帧一致，晚早窗差均为0；状态方差虽随Adapt Fisher变化，但没有转化为可见的参数更新或BER差异。因此固定未训练的高方差先验没有显示在线增量，也没有足够证据支持扩大这组固定参数。下一步不是继续调P0/Q，而是沿用离线轨迹内循环接口，固定rank-1 Fisher方向，通过离线未来Data外层目标学习跨SNR共享的初始方差、过程噪声和Pilot信息缩放；先验证梯度、拒绝回滚、投影一致性与信息隔离，再决定是否做新的session评估。

## 判断与下一项

这四帧不能证明状态空间机制有效或无效。它揭示了当前固定单步上限0.05过大：首个接受候选即耗尽累计半径，遮蔽了跨帧不确定度机制。下一项只把A/B两臂共同的单步更新上限改为0.005，其余状态超参数、方向、Reward规则、信道和观测资源固定；在新session上做8帧开发筛查。若候选接受仍很少或状态臂不优于同预算GN，停止当前rank-1状态滤波，不继续扩大该参数化。

## 复现

```powershell
python scripts/prototype_head_rank1_adapt.py --config configs/eme_long_memory_v2_rank1_state_filter_dev_v1.json --pretrained docs/results/full_head_softber_train32_v1_20261106/checkpoint/model_best.pt --basis-seed 78001 --seed-start 76204 --num-seeds 1 --frames 4 --device cuda --branches frozen,fisher_rank1_gn,fisher_rank1_state_filter --output <输出JSON>
```

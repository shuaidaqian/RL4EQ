# Soft Data 方差先导（AR(1)，10 × 1,024 帧）

## 状态

- 两臂均使用旧主 checkpoint `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，正常神经解码器路径（`decoder: neural`）；不训练或选择 P12 checkpoint。
- 四个 SNR：0、5、10、15 dB；10 个新 seed：240002–240011；每 seed、每 SNR 连续 1,024 帧。
- 同 seed 两臂共享接收波形、acquisition 估计与初始 tail，递归状态随后按各自策略自然分化。
- 两臂均使用 `process_noise=0` 与已冻结的 AR(1) 动态参数文件，SHA-256 为 `392cae39e89b9ffdd564a77bf3c3ebf852200c81bec43edcfdf7bc690e5a2e29`。
- 使用因果时序：本帧 soft Data 只影响后续帧。
- 仅在全部 10 个 seed 完成后，汇总每个 SNR 的 session 级配对绝对/相对 BER 差方差及跨 SNR 协方差；不做效果、伤害或假设检验，不按中途 BER 加 seed，不自动补 seed。
- 若运行中断，保留不完整状态并报告；不得把该先导写成“soft Data 有效/无害”。

## 资源上限

本先导按已对齐的 24 GPU 小时暂定总上限执行。因短时 AR(1) 计时外推可能不准，实际运行到达上限前 30 分钟仍未完成时，安全中断并标注 incomplete；保留已完成记录，不续跑、不补 seed，先报告实际 GPU/墙钟时间与完整性。

## 复现协议

完整机器可读协议见 [`continual_v2_soft_data_variance_pilot_10x1024_20261122.json`](../../protocols/continual_v2_soft_data_variance_pilot_10x1024_20261122.json)。执行命令使用 `scripts/compare_continual_variants.py` 的流式落盘和 128 帧检查点；开始前通过 seed registry 登记，记录 checkpoint、动态参数、配置和源码哈希；完成后对原始逐帧数据做完整性与波形配对审计。

## 运行记录（2026-11-22）

- 20:55–21:00 的三次启动均在接收循环开始前被 seed registry 状态门禁拒绝：启动包装脚本重复调用 `reserve_seeds` 把登记从 `reserved` 改为 `running`，而 `compare_continual_variants.py` 只接受 `reserved` 自行转为 `running`。三次均未生成 `variance_pilot.json`/`.jsonl`，未解码任何帧，种子未消耗。
- 上述失败的状态文件归档于 [`preflight_failed_launch_attempts/`](preflight_failed_launch_attempts/)。
- 包装脚本已移除重复登记；确认无输出后，将同一登记协调回 `reserved`（实验名、种子、purpose 均不变，未替换种子），由规范脚本重新登记并启动。
- 该先导仅为方差/协方差估计，不构成 BER 有效性或无害结论。

# 全 head Pilot/Data 更新方向事后诊断

`head_gradient_diagnostic.json` 由现有 `scripts/diagnose_head_transfer.py` 产生，覆盖普通续训和 trajectory 续训 checkpoint、同一开发 session 76120/76121、四个 SNR、每点每 session 8帧。所有 Data 梯度、Data BCE/BER 和候选 Data 改善均在候选生成并按 Pilot 规则接受判定后计算，只用于离线诊断。该脚本逐帧从 checkpoint 起点构造单帧候选，不复现跨帧已提交参数的累计状态；因此它解释局部更新机制，不替代此前的真实持续轨迹 A/B 评估。

两种 checkpoint 的结果几乎相同：

| SNR | Adapt/Data梯度余弦均值 | 候选平均 Data BCE 改善 | 接受数/32 | 候选BER收益 | 接受后的有害候选 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 约 +0.746 | 约 +2.09e-4 | 30 | 两种 checkpoint 分别 -8.14e-5、+8.14e-5 | 分别1、2 |
| 5 dB | 约 +0.650 | 约 +4.52e-5 | 18 | 0 | 0 |
| 10 dB | 约 -0.011 | 约 +1.83e-7 | 0 | 0 | 0 |
| 15 dB | 约 -0.0075 | 约 +4.59e-7 | 2 | 0 | 0 |

每SNR只有两个session，置信区间过宽且不适合作确认。解释上，0/5 dB局部 Adapt 梯度与 Data 梯度有正向一致性，候选也降低了连续 Data BCE；所以当前失败不能简单归因于“Pilot 梯度完全指错”。但 BER 收益接近零并随 checkpoint/session 改变符号，且0 dB存在被 Reward 接受的有害 Data 候选，显示短窗离散误码及 Pilot 门控不能稳定保证检测收益。10/15 dB的梯度方向近零、候选很少被接受且 Data BCE变化极小，表明高SNR缺少足以形成可观收益的更新信号；未发生 BER 翻转不等于更新有效。

这支持下一步把问题拆成两个独立机制：低SNR下校准Reward Pilot对连续Data风险的筛选能力，并检查跨帧参数累计/有害更新；高SNR下增加合法Adapt Pilot信息利用效率或缩小可更新子空间，但要先证实可辨识信号。Data指标仅用于此类离线机制诊断，绝不能用于在线接受规则。

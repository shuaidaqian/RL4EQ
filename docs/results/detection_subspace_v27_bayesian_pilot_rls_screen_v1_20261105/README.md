# v27来源上的低维Pilot贝叶斯/RLS机制筛查

## 范围与公平比较

这是一项独立原型机制筛查，**来源是v27的B_post_adapt弱开发checkpoint**，不代表当前离线性能最佳模型，也不能作为强离线模型在线增益证据。未改正式入口、训练流程、主配置、信道或资源协议。固定v27 checkpoint、同一批v27训练/选择/evaluation缓存、同一接收波形与2维在线参数，对比单帧阻尼GN和持久低维Pilot Bayes/RLS；两种在线分支均从同一离线系数起点开始，并使用v27原验证Pilot门控、0.5离线锚点收缩和源点回退。

在线候选只读当前帧合法Adapt Pilot波形与标签；当前验证Pilot只做候选门控；Data标签在两种算法的候选、接受及回滚全部结束后才读取评分。RLS超参先在训练角色预筛，再仅在选择角色Pilot验证BCE上作最终选择；没有使用训练/选择/evaluation Data标签调参，也没有用v27 eval结果改超参。

## 更新公式

在当前上下文`z_t`对Adapt Pilot logits构造Jacobian `J_t`，记`p=sigmoid(logit/temperature)`、`W=diag(p(1-p))`、Pilot标签为`y`。采用逐帧平均似然的局部Laplace/RLS近似：

```text
P_pred = P_(t-1) / lambda
g_t = J_t^T (p - y) / N
F_t = J_t^T W J_t / N
delta = -(P_pred^-1 + F_t + 2*prox*I)^-1
         [g_t + 2*prox*(z_t - z_origin)]
P_t = (P_pred^-1 + F_t)^-1   # 仅候选通过门控后保留
```

这里按每帧平均对数似然归一化，将一帧视为一个批Pilot观测，再跨帧累积信息。该标度避免把相关Pilot符号错误视为独立样本的精确Bayes后验；它是帧归一化的经验Fisher递推。候选经过和GN相同的逐步半径、总半径、验证Pilot BCE/硬BER守门、锚点收缩和初始点回退。当前先验协方差与遗忘因子为`lambda=0.98`、`P0=10I`。

训练Pilot预筛使用seed `121100/121101`（每seed覆盖四个SNR、每段9帧），共9组`lambda∈{0.98,0.995,1.0}`、`P0∈{0.1,1,10}`。按训练Pilot验证BCE取Top-3，再在全部选择seed `121200/121201`、四SNR、每段9帧上以验证Pilot保留后BCE选出最终超参。最终选择`lambda=0.98, P0=10`。网格与分角色指标见[`tuning_pilot_only.json`](tuning_pilot_only.json)。

## v27 eval cache筛查结果

留出缓存为seed `121700/121701/121702`，每seed/SNR 32帧。窗口为4个连续8帧块；增益定义为`BER_frozen - BER_online`，正值表示Online更好。统计CI以3个seed/session为单位，不能把帧或bit当作独立样本。

| SNR | 方法 | Overall增益（95%区间） | 趋势斜率/帧（95%区间） | 末窗−首窗（95%区间） | 接受/候选/回源 |
|---:|---|---:|---:|---:|---:|
| 0 dB | Bayes/RLS | -1.36e-05 [-1.30e-04, +1.03e-04] | -3.39e-06 [-1.88e-05, +1.20e-05] | -1.09e-04 [-5.75e-04, +3.58e-04] | 37/49/25 |
| 0 dB | 单帧GN | +4.07e-05 [-3.24e-04, +4.05e-04] | -7.46e-06 [-5.91e-05, +4.42e-05] | -1.63e-04 [-1.23e-03, +9.07e-04] | 40/48/22 |
| 5 dB | Bayes/RLS | 0 [-1.01e-04, +1.01e-04] | +1.36e-06 [-1.49e-05, +1.76e-05] | 0 [-4.04e-04, +4.04e-04] | 31/39/18 |
| 5 dB | 单帧GN | +5.43e-05 [-6.25e-05, +1.71e-04] | 约0 [-1.01e-05, +1.01e-05] | -1.09e-04 [-3.42e-04, +1.25e-04] | 33/41/21 |
| 10 dB | Bayes/RLS | 0 | 0 | 0 | 8/13/1 |
| 10 dB | 单帧GN | 0 | 0 | 0 | 17/22/4 |
| 15 dB | Bayes/RLS | 0 | 0 | 0 | 0/0/0 |
| 15 dB | 单帧GN | 0 | 0 | 0 | 0/0/0 |

RLS没有优于单帧GN的证据。0 dB Online均值略差于Frozen且末窗收益没有增长；5 dB平均BER与Frozen近乎相同、趋势区间跨零；10/15 dB没有可用的增长信号。单帧GN自身的末窗−首窗也为负或接近零，所以本次结果不支持“随着传输帧数增加收益变大”。RLS在10 dB的候选/接受数明显低于GN，但没有减少0/5 dB有害更新到足以形成BER增益。完整逐帧与session/window趋势见[`summary.json`](summary.json)和[`frame_metrics.jsonl.gz`](frame_metrics.jsonl.gz)。

## 限制与下一步

这是用v27弱来源checkpoint和仅3个短eval session做的机制筛查；32帧×3 session不能验证长时持续趋势或统计确认，也不能外推到当前离线训练的最佳checkpoint。RLS第一版使用局部Jacobian、逐帧归一化Fisher和v27原硬验证门控，结果只否定这组固定超参/实现，没有排除所有贝叶斯、RLS或信赖域方法。下一步先将相同原型用于用户指出的强离线模型 checkpoint，并在同一长会话和同一物理跟踪协议下做Frozen/GN/RLS对照；如用户目标要求帧数增益增长，应延长到预先定义的多窗口会话，并按独立session估计趋势。正式确认仍未启动。

## 复现命令

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' scripts/prototype_bayesian_pilot_v27.py --output-dir docs/results/detection_subspace_v27_bayesian_pilot_rls_screen_v1_20261105 --device cuda
```

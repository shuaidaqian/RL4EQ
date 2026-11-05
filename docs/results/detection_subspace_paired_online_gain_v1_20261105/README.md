# 第十四轮：离线外循环显式优化Online相对Frozen的收益

## 研究问题与单因素

前几轮的B臂降低Online平滑BER，但没有稳定优于同一网络的Frozen输出；原外层主要优化Online绝对误码，未显式惩罚Online相对Frozen落后。本轮固定v11的时序卷积基、8维在线系数、Adapt温度2、每帧3步Adam、8帧历史、投影/锚点收缩、验证Pilot硬门控、四臂、训练与选择seed、四SNR及全部信道/接收资源。唯一改变是离线外层与checkpoint选择目标：对同一帧/下一帧及同一Data分别评分Frozen和Online，优化

`L_pair = L_frozen + L_online + softplus(L_online - L_frozen)`

其中`L`均为固定温度T=1的平滑BER。该目标保持降低两臂绝对损失的梯度，同时额外推动Online低于Frozen。它只用于离线训练/选择；在线仍只接收Adapt Pilot和验证Pilot，Data标签不进入在线控制。该候选对当前冻结模型的相对增益仍是开发主张，不是正式确认。

清单：[`detection_subspace_long_v12_paired_online_gain_v1.json`](../../../configs/audits/detection_subspace_long_v12_paired_online_gain_v1.json)。模型与训练配置没有修改信道字段，`channel_changes=false`。独立inspection选中8维；普通监督作用检查确认子空间/系数可以改变输出。A/B各64步，训练seed、selection seed与评估seed按清单隔离；评估为四SNR×3个既有开发seed×32帧×2臂，共768条配对记录，不是独立确认。

## 结果

增益定义为`BER_frozen - BER_online`，正值表示在线更好。

| SNR | A：普通监督Online增益（95%区间） | B：配对增益目标Online增益（95%区间） | B−A交互 | B接受数/候选接受数 |
|---:|---:|---:|---:|---:|
| 0 dB | `+1.36e-5` `[-4.48e-5,+7.19e-5]` | `+1.36e-5` `[-4.48e-5,+7.19e-5]` | 约`0` | `44/44` |
| 5 dB | `约0` `[-1.01e-4,+1.01e-4]` | `-1.36e-5` `[-7.19e-5,+4.48e-5]` | `-1.36e-5` | `42/42` |
| 10 dB | `0` `[0,0]` | `+1.36e-5` `[-4.48e-5,+7.19e-5]` | `+1.36e-5` | `2/3` |
| 15 dB | `0` `[0,0]` | `0` `[0,0]` | `0` | `0/0` |

B臂正向的0和10 dB点估计分别来自三个开发会话中单个会话约少1个Data bit；5 dB有一个会话多约1个错误。配对区间全部跨零，四SNR没有形成一致的在线优越证据；15 dB没有更新。B checkpoint选择目标`0.840494`略差于A的`0.840394`。B在线子空间更新约44–47 ms/帧，不含物理状态跟踪、acquisition和端到端接收。

## 结论与下一步

显式加入Frozen/Online配对增益项没有转成可重复BER改善，因此本目标版本失败，不作为默认训练目标，核心研究目标仍未完成。它排除了“只要在离线目标中写入Online-Frozen差值就足以带来在线增益”的简单解释。

随后对B checkpoint和固定8维表示运行了只读边界可达性分析，细节见本目录[`data_boundary_reachability.json`](data_boundary_reachability.json)及[`诊断脚本`](../../../scripts/diagnose_subspace_data_boundary_reachability.py)。当前3个开发会话合计，单个0.15半径内理论上可触及的既有Data错误只占0/5/10/15 dB的`0.56%/1.86%/3.60%/4.88%`；按每帧3步的最大位移上界0.45放宽后也只有`1.81%/5.67%/18.00%/21.95%`。这只是当前线性子空间局部可达上界，不表示Pilot能找到该方向。真实Pilot候选方向在既有错误符号上增加边距的比例为`29.96%/33.12%/52.80%/43.90%`，但同时减少边距的比例约为`70.04%/66.88%/47.20%/56.10%`；实际候选仅翻转`4/1/0/0`个错误，门控保留后总错误为`12766/2471/250/41`，与更新前`12766/2470/250/41`基本相同。

诊断说明瓶颈不是单一门控：当前可训练子空间对很多错误符号只在较大系数变化下才有机会翻转，Adapt候选又同时帮到和伤到不同Data错误。下一单因素不再调Pilot阈值或Adam步长；先试**离线Frozen错误条件化Online目标**：按冻结网络的Data错误/正确符号分桶，训练在线更新把原有错误推过判决边界，同时维持原本正确符号的边距。Data分桶只在离线外循环使用；部署更新仍仅依赖Adapt/验证Pilot。该方向先用固定开发集做四臂机制验证；若没有相对Frozen的正增益，就重设计可适配表示。正式确认仍不启动。

## 复现与归档

定向测试：`32 passed`。v14代码改动后的全量测试：`933 passed in 370.85s`。全量测试后新增的边界诊断脚本已在本机直接运行并生成归档JSON；提交前对新增脚本执行语法检查，`git diff --check`通过。

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests/test_detection_subspace.py
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method subspace_audit --audit-manifest configs/audits/detection_subspace_long_v12_paired_online_gain_v1.json --output-dir logs/detection_subspace_inspection_v12_paired_online_gain_v1_20261105 --device cuda
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --stage detection_subspace --subspace-manifest configs/audits/detection_subspace_long_v12_paired_online_gain_v1.json --inspection-dir logs/detection_subspace_inspection_v12_paired_online_gain_v1_20261105 --save-dir logs/detection_subspace_training_v12_paired_online_gain_v1_20261105 --device cuda
```

归档包含inspection、清单、摘要及压缩逐帧轨迹。轨迹SHA256：`bbfe9a02eab8303869d3371b1b38f5be12d1479e8cc16bc7435ea44a3fd5896a`。完整训练日志、checkpoint及特征缓存保留在本机忽略的`logs/`目录，不提交大体积缓存。

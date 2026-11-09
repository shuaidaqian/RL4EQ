# 参考文献库（物理在线方法调研，2026-11-09）

本目录为 `physical-online-methods-search` 分支的非神经、非 RL 在线方法调研文献。
PDF 以论文标题命名；未获取的付费文献在下方列出，引用前请自行通过机构订阅核对。

## 已下载

| 文件 | 作者/年份 | 主题 | 来源 |
|---|---|---|---|
| `Frequency-domain equalization of single carrier transmissions over doubly selective channels.pdf` | Hong Liu, 2007（导师 Schniter） | 单载波 + 软输入 Kalman + 长延迟扩展 | OhioLINK ETD |
| `Joint Channel Estimation and Data Detection in MIMO-OFDM Systems - A Sparse Bayesian Learning Approach.pdf` | Prasad, Murthy & Rao, IEEE TSP 2015 | SBL 联合信道估计/数据检测 | IISc 作者页 |
| `Sparse Bayesian Learning for Joint Channel Estimation and Data Detection in OFDM Systems.pdf` | Ranjitha Prasad, PhD thesis 2015 | J-SBL / KSBL / 递归 J-SBL 完整推导 | IISc 作者页 |
| `Joint Frequency Domain Channel Estimation and Equalization Based on Expectation Propagation for Single Carrier Transmissions.pdf` | Şahin, Cipriano, Poulliat & Boucheret, ICASSP 2020 | 单载波 EP 联合信道估计/均衡 | HAL |
| `Joint Channel Estimation and Turbo Equalization of Single-Carrier Systems over Time-Varying Channels.pdf` | arXiv 2305.09226 | 单载波 EP 消息传递/turbo 均衡 | arXiv |
| `Joint Data QR-Detection and Kalman Estimation for OFDM Time-varying Rayleigh Channel Complex Gains.pdf` | Hijazi & Ros, IEEE TCOM 2010 | QR 检测 + AR Kalman | HAL |
| `Identifying Time-Varying Channels with Aid of Pilots for MIMO-OFDM.pdf` | Tang & Leus, EURASIP JASP 2011 | BEM/导频辅助时变信道估计（开放版，替代 TSP 2007） | SpringerOpen |

## 未获取（付费，需机构订阅）

| 文献 | 主题 | 备注 |
|---|---|---|
| Barbu et al., *OFDM Receiver for Fast Time-Varying Channels Using Block-Sparse Bayesian Learning*, IEEE TVT 2016, DOI `10.1109/TVT.2016.2554611` | 块稀疏 SBL + BEM | Aalborg VBN 有记录但无开放 PDF |
| Nissilä & Pasupathy, *Adaptive iterative detectors for phase-uncertain channels via variational bounding*, IEEE TCOM 2009, DOI `10.1109/TCOMM.2009.03.070068` | VB 联合相位/检测 | OuluREPO 直链 403 |
| Nissilä, PhD thesis, *Iterative receivers for digital communications via variational inference and estimation*, Oulu 2008 | VB/BEM/AR 信道 | OuluREPO 直链 403 |
| Dagres & Polydoros, *Decision-directed least-squares phase perturbation compensation in OFDM systems*, IEEE TWC 2009, DOI `10.1109/TWC.2009.081420` | DD 最小二乘相位补偿 | UoA Pergamos 无公开 PDF |
| Muralidhar & Li, *A low-complexity Kalman approach for channel estimation in doubly-selective OFDM*, IEEE SPL 2009 | 低复杂度 Kalman | 未找到开放版 |
| Pedrosa et al., *Channel Tracking and Equalization for UAV Communications*, Springer 2022, DOI `10.1007/978-3-031-03880-8_1` | SC-FDE + IB-DFE + EKF | 书章，付费 |
| Pham, Liang & Nallanathan, *Doubly iterative receiver for block transmissions with EM-based channel estimation*, IEEE TWC 2009, DOI `10.1109/TWC.2009.071416` | EM + BI-GDFE | 未找到开放版 |
| Prasad & Murthy, *Bayesian Learning for Joint Sparse OFDM Channel Estimation and Data Detection*, GLOBECOM 2010 | SBL 联合估计 | 内容已由 Prasad 2015 学位论文覆盖 |
| Wei, Zhang & Liu, *Expectation-Propagation Based Low-Complexity Channel Estimation for Massive MIMO Systems*, ICC 2017 | EP/SBL 低复杂度 | 付费；NVIDIA Sionna 有 `SBLChannelEstimator` 实现 |

## 使用边界

- PDF 仅用于本地阅读与引用核对，不参与代码运行。
- 本目录文献为摘要级/作者版核验；正式引用前请核对 DOI、卷期页码。

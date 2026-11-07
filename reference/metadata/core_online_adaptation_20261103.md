# 离线训练与在线微调均衡：7篇核心参考文献

整理日期：2026-11-03。对应本次研究路线分析中的文献[1]–[7]。已有全文直接复用，缺失全文从NIST或arXiv补齐。

正式引用以DOI对应的出版信息为准；arXiv文件是作者公开版本，不保证与出版社排版版逐字相同。文件名中的预印本年份不替代正式出版年份。

|编号|正式出版标题与年份|全文|文本|研究用途|
|---|---|---|---|---|
|1|Radar Studies of the Moon（1965）|[PDF](1965-evans-radar-studies-moon.pdf)|[TXT](1965-evans-radar-studies-moon.txt)|EME 长回波、月面散射、延迟和多普勒的原始物理背景。|
|2|Learning to Demodulate From Few Pilots via Offline and Online Meta-Learning（2021）|[PDF](1908.09049-park2021-few-pilots-meta-learning.pdf)|[TXT](1908.09049-park2021-few-pilots-meta-learning.txt)|少 Pilot 下的离线元学习、在线快速适配和低维 context。|
|3|Online Meta-Learning for Hybrid Model-Based Deep Receivers（2023）|[PDF](2203.14359-raviv-meta-learning-hybrid-receivers.pdf)|[TXT](2203.14359-raviv-meta-learning-hybrid-receivers.txt)|物理模型辅助接收、预测式元学习、当前和历史任务配合。|
|4|ViterbiNet: A Deep Learning Based Viterbi Algorithm for Symbol Detection（2020）|[PDF](1905.10750-shlezinger2020-viterbinet.pdf)|[TXT](1905.10750-shlezinger2020-viterbinet.txt)|保留接收算法结构，仅学习模型相关的局部计算。|
|5|Bayesian Active Meta-Learning for Reliable and Efficient AI-Based Demodulation（2022）|[PDF](2108.00785-cohen2022-bayesian-active-meta-learning.pdf)|[TXT](2108.00785-cohen2022-bayesian-active-meta-learning.txt)|少样本适配、不确定性校准和主动元训练。|
|6|Asynchronous Online Adaptation via Modular Drift Detection for Deep Receivers（2025）|[PDF](2407.09134-uzlaner2024-asynchronous-drift-adaptation.pdf)|[TXT](2407.09134-uzlaner2024-asynchronous-drift-adaptation.txt)|基于漂移检测决定何时更新、更新哪个模块，作为非 RL 调度参考。|
|7|Online Learning of Modular Bayesian Deep Receivers: Single-Step Adaptation With Streaming Data（2026）|[PDF](2511.06045-gusakov2025-online-bayesian-receiver.pdf)|[TXT](2511.06045-gusakov2025-online-bayesian-receiver.txt)|把在线神经参数更新组织为流式贝叶斯跟踪。|

## 来源与版本说明

1. **Radar Studies of the Moon**
   - DOI：[10.6028/jres.069d.195](https://doi.org/10.6028/jres.069d.195)。
   - 下载来源：https://nvlpubs.nist.gov/nistpubs/jres/69D/jresv69Dn12p1637_A1b.pdf。
   - 版本说明：NIST 公开的原刊扫描版；扫描文字抽取可能存在 OCR 误差。
   - 核验：PDF文件头、可解析页数和首页主题词通过；23页，30199648字节。

2. **Learning to Demodulate From Few Pilots via Offline and Online Meta-Learning**
   - DOI：[10.1109/TSP.2020.3043879](https://doi.org/10.1109/TSP.2020.3043879)。
   - 下载来源：https://arxiv.org/pdf/1908.09049v3。
   - 版本说明：正式出版年份为2021，DOI中2020不是本文采用的出版年份；PDF为作者公开版本。
   - 核验：PDF文件头、可解析页数和首页主题词通过；15页，13856921字节。

3. **Online Meta-Learning for Hybrid Model-Based Deep Receivers**
   - DOI：[10.1109/TWC.2023.3241841](https://doi.org/10.1109/TWC.2023.3241841)。
   - 下载来源：https://arxiv.org/pdf/2203.14359。
   - 版本说明：复用本目录既有公开全文；文中短记忆和编码反馈条件不能直接等同于当前长回波系统。
   - 核验：PDF文件头、可解析页数和首页主题词通过；32页，3194473字节。

4. **ViterbiNet: A Deep Learning Based Viterbi Algorithm for Symbol Detection**
   - DOI：[10.1109/TWC.2020.2972352](https://doi.org/10.1109/TWC.2020.2972352)。
   - 下载来源：https://arxiv.org/pdf/1905.10750v2。
   - 版本说明：PDF为作者公开版本；原始 Viterbi 状态空间不能直接扩展到116符号记忆。
   - 核验：PDF文件头、可解析页数和首页主题词通过；28页，1037815字节。

5. **Bayesian Active Meta-Learning for Reliable and Efficient AI-Based Demodulation**
   - DOI：[10.1109/TSP.2022.3220035](https://doi.org/10.1109/TSP.2022.3220035)。
   - 下载来源：https://arxiv.org/pdf/2108.00785v3。
   - 版本说明：公开PDF标题为 Bayesian Active Meta-Learning for Few Pilot Demodulation and Equalization，与正式出版标题不同。
   - 核验：PDF文件头、可解析页数和首页主题词通过；16页，6792703字节。

6. **Asynchronous Online Adaptation via Modular Drift Detection for Deep Receivers**
   - DOI：[10.1109/TWC.2025.3533959](https://doi.org/10.1109/TWC.2025.3533959)。
   - 下载来源：https://arxiv.org/pdf/2407.09134。
   - 版本说明：复用既有文件；文件名中的2024是预印本年份，正式出版年份为2025。
   - 核验：PDF文件头、可解析页数和首页主题词通过；13页，1803862字节。

7. **Online Learning of Modular Bayesian Deep Receivers: Single-Step Adaptation With Streaming Data**
   - DOI：[10.1109/TSP.2026.3697255](https://doi.org/10.1109/TSP.2026.3697255)。
   - 下载来源：https://arxiv.org/pdf/2511.06045。
   - 版本说明：复用既有文件；文件名中的2025是预印本年份，正式出版元数据年份为2026。公开版本的投稿状态可能早于正式出版状态。
   - 核验：PDF文件头、可解析页数和首页主题词通过；15页，7024792字节。

## 阅读顺序

先读[2]明确“离线学会快速适配”的任务，再读[3][4]理解模型辅助接收结构，随后读[5][7]处理少样本与持续跟踪的不确定性，[6]用于比较无需RL的更新调度，[1]用于核对EME物理背景。

## 配套文件

- [下载与SHA-256清单](core_online_adaptation_20261103_manifest.json)：逐篇记录来源、文件校验和、页数与正式出版元数据。
- [BibTeX引用文件](core_online_adaptation_20261103.bib)：以正式出版信息生成，PDF链接指向本地公开全文。

本次索引独立于目录中既有RL、MIMO、OFDM等历史资料；收录相关文献不表示恢复这些研究路线。

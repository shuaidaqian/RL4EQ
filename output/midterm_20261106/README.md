# 第二研究点材料：离线训练 + 在线贝叶斯 CIR 适配

更新：2026-11-07

研究点二完整表述为「**离线训练适应信道，在线微调做均衡**」：
离线用仿真数据训练 Pilot 条件化的神经块均衡器；部署时冻结网络权重，
仅用合法观测递推贝叶斯 CIR 后验，使均衡器持续匹配当前信道。

---

## 一、交付物（直接看这些）

| 文件 | 说明 |
|---|---|
| `中期答辩_李诚谦_20261106.pptx` / `.pdf` | 中期答辩 PPT，18 页，含演讲者备注 |
| `中期报告_李诚谦_20261106.docx` / `.pdf` | 书面中期报告，12 页，约 8700 字 |
| `论文初稿_第二研究点_v1.pdf` | 期刊论文初稿，18 页，约 8400 字 |
| `contact_sheet.png` | 18 页 PPT 缩略图 |

## 二、源码与源文件

| 路径 | 内容 |
|---|---|
| `paper/` | 论文 LaTeX 工程。打开 `paper/main.tex` 用 XeLaTeX 编译 |
| `paper/sections/` | 按章节拆分的 .tex |
| `figures/` | 全部生成图（`brand/` 为校徽等素材，`paper/` 为论文原图） |
| `render/` | PPT 逐页导出 PNG，命名 `slide01.png` … `slide18.png` |
| `midterm_report_v2.md` | 书面报告的 Markdown 底稿 |
| `paper_draft_v1.md` | 论文的 Markdown 底稿 |

## 三、生成脚本

| 脚本 | 作用 |
|---|---|
| `make_figures.py` | 生成全部数据图（从 `docs/results/` 的 JSON 直接读数） |
| `make_ppt.py` | 生成答辩 PPT |
| `build_report.py` | 由 md 底稿生成 docx |
| `convert_md2tex.py` | 由论文 md 生成 LaTeX 各章节 |
| `qa_ppt.py` | PPT 结构自检（越界、备注、图片分辨率） |
| `render_check.py` | 调用本机 PowerPoint 逐页导出 PNG |
| `export_pdf.py` | PPT 导出 PDF |
| `style.py` | 配色、字体、版式的统一定义 |

**重新生成需要切换到 `physical-bayes-cir-v1` 分支的工作树**
（`D:/Research/RL4EQ-worktrees/physical`），
因为脚本读取的 `docs/results/bayesian_cir_confirmation_v1_20261106/`、
`docs/results/bayesian_cir_diag_v1_20261106/`、`docs/results/offline_saturation_20261106/`
只存在于该分支。当前 `D:\Research\RL4EQ` 检出的是 `codex/physical-state-tracking`，
不含这些归档，脚本在此目录不能直接运行。

生成后在物理工作树里重新生成，再同步回本目录即可。

## 四、关键结果速查

### 离线训练（3 seed × 60 帧主矩阵）

| SNR | 离线神经均衡器 | CFO+DD-Phase DFE-RLS | 相对降低 |
|---:|---:|---:|---:|
| 0 dB | 22.19% | 44.05% | 49.6% |
| 5 dB | 5.96% | 18.30% | 67.4% |
| 10 dB | 1.06% | 6.91% | 84.7% |
| 15 dB | 0.216% | 4.37% | 95.1% |

### 在线适配独立确认（seed 100001–100010，10 session × 128 帧）

| SNR | Frozen BER | Bayes BER | ΔBER | 单侧 95% 下界 | Holm p |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.2081 | 0.1713 | +0.03680 | +0.03029 | 1e-5 |
| 5 dB | 0.0479 | 0.0399 | +0.00803 | +0.00627 | 2e-5 |
| 10 dB | 0.00623 | 0.00426 | +0.00197 | +0.00146 | 6e-5 |
| 15 dB | 0.000985 | 0.000724 | +0.000260 | +0.000149 | 1e-3 |

- H1（四 SNR 等权平均 +0.011767，下界 +0.010188，p=1.26e-7）通过
- H2（逐 SNR 下界 > 0、10/10 session 无负向）通过
- H3（增益随帧增长）5/10/15 dB 通过；**0 dB 斜率 -0.001238/bin，不通过**
- CIR 的 NMSE 下降 67%–75%

## 五、信息边界（写作与汇报时必须保持）

- 在线只读：已接收波形、Adapt Pilot 已知符号、模型 soft 输出
- **Data 段的波形**参与均衡与无标签更新（判决反馈），Data 后验用的是模型软判决
  `tanh(logit/2)` 作为伪观测，噪声项 `R_t = σ² + Σ|h_l|²Var(x̂_{t-l})` 使不可靠符号自动降权
- **Data 段的发送标签**禁止进入任何在线更新、选择、验收或阈值设定，只用于事后评分
- Reward Pilot 只在动作之后用于验收，不回流内循环

## 六、尚未完成

- 逐帧延迟与 GPU 显存未插桩；论文 4.4 节的时间数字来自单独 profile，需与实测核对
- 离线训练的原始命令、环境与随机种子未随 checkpoint 保存，复现链不完整
- 与传统接收机的公平对照仍是 3×16 开发规模，不是确认级证据
- 参考文献为手写 GB/T 7714 风格，投稿前需换成目标期刊模板

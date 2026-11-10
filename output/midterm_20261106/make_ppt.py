"""生成中期答辩 PPT。

版式全部手绘，不使用 Office 默认主题。
运行：python output/midterm_20261106/make_ppt.py
"""

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

import style
from style import FIGDIR, INK, MUTED, NAVY, BAYES

HERE = Path(__file__).resolve().parent
PAPER = FIGDIR / "paper"
BRAND = FIGDIR / "brand"
OUT = HERE / "中期答辩_李诚谦_v4.pptx"

SW, SH = 13.333, 7.5
MX = 0.75                      # 左右页边距
CW = SW - 2 * MX               # 内容宽 11.833

FONT = "微软雅黑"
LINE_CLR = "#DCD6CC"
PANEL = "#F2F4F7"
ALERT = "#B4472F"
GOOD = "#4F7C5B"
TEAL = "#2E6F8E"
BLUE = "#3E7CB1"
FROZEN = "#7A8FA6"


def C(hexstr):
    return RGBColor.from_string(hexstr.lstrip("#"))


# ---------------------------------------------------------------- 基础绘图


def set_run(run, size, bold=False, color=INK, font=FONT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = C(color)
    run.font.name = font
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def text(slide, s, x, y, w, h, size=13, bold=False, color=INK,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line=1.35, space_after=0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor

    paras = s.split("\n")
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = line
        para.space_after = Pt(space_after)
        run = para.add_run()
        run.text = p
        set_run(run, size, bold, color)
    return box


def rect(slide, x, y, w, h, fill=None, line_color=None, line_w=0.75,
         shape=MSO_SHAPE.RECTANGLE, radius=None):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = C(fill)
    else:
        shp.fill.background()
    if line_color:
        shp.line.color.rgb = C(line_color)
        shp.line.width = Pt(line_w)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    return shp


def hline(slide, x1, y, x2, color=LINE_CLR, width=0.75):
    ln = slide.shapes.add_connector(1, Inches(x1), Inches(y), Inches(x2), Inches(y))
    ln.line.color.rgb = C(color)
    ln.line.width = Pt(width)


def vline(slide, x, y1, y2, color=LINE_CLR, width=0.75):
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y1), Inches(x), Inches(y2))
    ln.line.color.rgb = C(color)
    ln.line.width = Pt(width)


def set_alpha(pic, opacity_pct):
    """设置图片不透明度（0–100 表示保留的不透明比例）。"""
    blip = pic._element.blipFill.find(qn("a:blip"))
    if blip is not None:
        blip.append(blip.makeelement(qn("a:alphaModFix"),
                                     {"amt": str(int(opacity_pct * 1000))}))


def pic_fit(slide, path, bx, by, bw, bh):
    """等比缩放并居中放入给定框。"""
    iw, ih = Image.open(path).size
    s = min(bw / iw, bh / ih)
    w, h = iw * s, ih * s
    return slide.shapes.add_picture(str(path), Inches(bx + (bw - w) / 2),
                                    Inches(by + (bh - h) / 2), Inches(w), Inches(h))


# ---------------------------------------------------------------- 页面骨架


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def chrome(slide, idx, total):
    """页脚：细线 + 左上角机构名 + 右下角页码。"""
    hline(slide, MX, 6.98, SW - MX)
    slide.shapes.add_picture(str(BRAND / "logo_round.png"), Inches(MX),
                             Inches(7.02), height=Inches(0.30))
    text(slide, "北京邮电大学 · 信息与通信工程学院　　硕士学位论文中期检查",
         MX + 0.42, 7.06, 9.0, 0.3, size=8.5, color=MUTED)
    text(slide, f"{idx:02d} / {total}", SW - MX - 1.2, 7.06, 1.2, 0.3,
         size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)


def page(prs, idx, total, kicker, title, note=""):
    slide = blank(prs)
    # 左侧琥珀短竖线
    rect(slide, MX, 0.52, 0.055, 0.60, fill=BAYES)
    if kicker:
        text(slide, kicker, MX + 0.20, 0.40, CW - 0.2, 0.26,
             size=9.5, color=MUTED)
        ty = 0.66
    else:
        ty = 0.52
    text(slide, title, MX + 0.20, ty, CW - 0.2, 0.62,
         size=24, bold=True, color=NAVY)
    if note:
        text(slide, note, MX, 1.22, CW, 0.30, size=11.5, color=MUTED)
    chrome(slide, idx, total)
    return slide


def speaker(slide, s):
    slide.notes_slide.notes_text_frame.text = s


# ---------------------------------------------------------------- 各页


def cover(prs, idx, total):
    slide = blank(prs)
    deco = slide.shapes.add_picture(str(BRAND / "main_building.png"),
                                    Inches(8.50), Inches(3.40), width=Inches(4.65))
    set_alpha(deco, 26)

    slide.shapes.add_picture(str(BRAND / "logo_horizontal.png"),
                             Inches(MX), Inches(0.66), width=Inches(2.60))
    text(slide, "信息与通信工程学院", MX, 1.50, 6.0, 0.28, size=10.5, color=MUTED)
    hline(slide, MX, 1.84, MX + 5.4, color=NAVY, width=1.2)

    text(slide, "宽带 EME 通信系统的信道建模", MX, 2.05, 11.5, 0.85,
         size=38, bold=True, color=NAVY)
    text(slide, "与基带波形设计关键技术研究", MX, 2.85, 11.5, 0.85,
         size=38, bold=True, color=NAVY)

    rect(slide, MX, 3.95, 0.9, 0.05, fill=BAYES)
    text(slide, "硕士学位论文中期检查报告", MX, 4.18, 8.0, 0.45,
         size=19, color=INK)

    info = [
        ("研究生", "李诚谦"), ("学号", "2024140313"),
        ("导师", "于赫"), ("时间", "2026 年 11 月"),
    ]
    x = MX
    for k, v in info:
        text(slide, k, x, 5.85, 1.1, 0.28, size=9.5, color=MUTED)
        text(slide, v, x, 6.12, 2.4, 0.34, size=13, color=INK)
        x += 2.55

    text(slide, f"共 {total} 页　·　汇报 15 分钟", SW - MX - 3.4, 6.95, 3.4, 0.28,
         size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)
    speaker(slide, "各位老师上午好，我是通信工程专业的李诚谦，导师是于赫老师。"
                   "今天汇报学位论文的中期进展，重点是两个研究点的完成情况与证据强度。")
    return slide


def slide_route(prs, idx, total):
    s = page(prs, idx, total, "研究全景",
             "一条主线：从月面散射物理，走到可在线自适应的接收机")
    pic_fit(s, FIGDIR / "fig_route.png", MX, 1.55, CW, 4.55)
    text(s, "研究点二的两步：离线把 BER 降一个数量级，在线把物理状态跟住",
         MX, 6.28, CW, 0.34, size=12, color=MUTED)
    speaker(s, "整个课题是一条链：月面散射物理决定信道，信道的长记忆决定接收机难点。"
               "研究点一已经把前半段做完；研究点二在解决后半段的在线适配问题。")
    return s


def slide_model(prs, idx, total):
    s = page(prs, idx, total, "研究点一 · 宽带 EME 信道建模",
             "统一积分模型保留窄波束下的月面空间差异")
    pic_fit(s, PAPER / "fig1.png", MX, 1.60, 6.55, 4.75)

    x = 7.62
    items = [
        ("分布式目标", "月球不是点目标，按波束照射区域离散成环带积分"),
        ("逐区域计权", "每环带考虑有效 RCS、入射角与天线增益"),
        ("可退化", "宽波束退化为平均 RCS 模型，窄波束保留空间差异"),
    ]
    y = 1.78
    for i, (h, body) in enumerate(items):
        rect(s, x, y + 0.06, 0.045, 0.30, fill=BAYES)
        text(s, h, x + 0.18, y, 4.3, 0.32, size=13, bold=True, color=INK)
        text(s, body, x + 0.18, y + 0.36, 4.3, 0.80, size=11, color=MUTED)
        y += 1.32

    text(s, "成果：《宽带 EME 通信信道的建模与特性分析》拟录用",
         x, 5.95, 4.4, 0.30, size=10.5, color=NAVY)
    speaker(s, "传统 EME 链路预算把月球当平均散射体。但高增益窄波束下，"
               "实际被照亮的月面区域形状和 RCS 分布会明显影响接收功率。"
               "这个统一积分模型的贡献，是让宽波束和窄波束落在同一个式子里。")
    return s


def slide_rcs(prs, idx, total):
    s = page(prs, idx, total, "研究点一 · 大尺度",
             "RCS 在月面分布并不均匀，平均模型在窄波束下会偏离",
             "左侧：月面 RCS 的空间分布　｜　右侧：不同波束宽度下的接收功率")
    pic_fit(s, PAPER / "fig2.png", MX, 1.85, 5.55, 4.45)
    pic_fit(s, PAPER / "fig3.png", 6.55, 1.85, 5.9, 4.45)
    text(s, "方向增益提升 ≠ 接收功率线性增长：照射面积与 RCS 分布同步变化",
         MX, 6.40, CW, 0.30, size=11.5, color=MUTED)
    speaker(s, "左边是 RCS 的空间分布，右边是对应的接收功率。"
               "结论是：提高天线增益时，被照亮的区域也在变，"
               "所以不能按点目标模型无限外推。")
    return s


def slide_delay(prs, idx, total):
    s = page(prs, idx, total, "研究点一 · 小尺度",
             "准镜面强回波 + 漫散射尾部，构成长达 116 符号的记忆")
    pic_fit(s, PAPER / "fig4.png", MX, 1.60, 6.4, 4.65)
    x = 7.45
    steps = [("准镜面分量", "早期集中的强回波"),
             ("漫散射分量", "长而衰减的时延尾部"),
             ("接收机含义", "有限记忆结构必然失配")]
    y = 1.95
    for i, (h, body) in enumerate(steps):
        text(s, h, x, y, 4.4, 0.30, size=12.5, bold=True, color=INK)
        text(s, body, x, y + 0.32, 4.4, 0.40, size=10.5, color=MUTED)
        if i < 2:
            text(s, "↓", x, y + 0.72, 0.4, 0.30, size=14, color=BAYES)
        y += 1.18
    text(s, "仿真：100 条多径，最大相对时延 10 ms", x, 5.62, 4.4, 0.30,
         size=10, color=MUTED)
    speaker(s, "信道被分成准镜面强回波和漫散射尾部两段。"
               "换算到符号域，记忆长度达到 116 个符号——这决定了后面接收机的设计。")
    return s


def slide_baseline(prs, idx, total):
    s = page(prs, idx, total, "研究点一 · 接收性能",
             "传统均衡在长记忆下饱和，多径而非频偏才是主瓶颈",
             "左：不同均衡器 BER　｜　右：FFT 与 MLE 频偏校正对比")
    pic_fit(s, PAPER / "fig5.png", MX, 1.85, 5.6, 4.45)
    pic_fit(s, PAPER / "fig6.png", 6.55, 1.85, 5.9, 4.45)
    text(s, "LMS / RLS / DFE / RAKE 改善趋于饱和　→　有限记忆与长时延失配",
         MX, 6.40, CW, 0.30, size=11.5, color=MUTED)
    speaker(s, "线性 LMS、RLS、判决反馈、RAKE 都试过，性能都趋于饱和。"
               "MLE 频偏校正比 FFT 稳健，但频偏不是主要矛盾——主要矛盾是长记忆多径。"
               "这直接引出第二研究点。")
    return s


def slide_offline(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 离线训练",
             "离线训练：神经均衡器把 BER 降到传统均衡的 5%–50%")
    pic_fit(s, FIGDIR / "v2" / "fig_offline_ber.png", MX, 1.62, 7.80, 4.50)

    x = 8.80
    items = [
        ("监督信号", "整帧 Data 标签只进入离线反向传播与 checkpoint 验证"),
        ("对照设置", "同一信道、同一 Pilot 预算，不使用神经网络"),
        ("结论", "离线均衡能力已明显优于传统基线，构成在线微调的起点"),
    ]
    y = 1.95
    for h, body in items:
        rect(s, x, y + 0.05, 0.045, 0.26, fill=BAYES)
        text(s, h, x + 0.18, y, 3.80, 0.30, size=12, bold=True, color=NAVY)
        text(s, body, x + 0.18, y + 0.33, 3.80, 0.85, size=10.2, color=MUTED)
        y += 1.42

    text(s, "主矩阵：3 seed × 60 帧；传统对照为合法 Pilot 驱动的 CFO/相位补偿接收机，不使用神经网络",
         MX, 6.34, CW, 0.30, size=9.8, color=MUTED)
    speaker(s, "第二个研究点分两步。第一步是离线训练：用仿真信道生成带标签的数据，"
               "监督训练一个 Pilot 条件化的神经均衡器。同一个信道、同样公平的 Pilot 条件下，"
               "它的 BER 比传统均衡器低 50% 到 95%。这是在线微调的起点。")
    return s


def slide_offline_arch(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 离线训练",
             "训练结构：Pilot 条件化的迭代展开网络")
    pic_fit(s, FIGDIR / "fig_offline_arch.png", MX, 1.62, CW, 3.90)

    kpis = [
        ("189,350", "网络参数"),
        ("全帧 BCE", "训练损失"),
        ("32 步", "训练预算，另含 8 次物理 warm start"),
    ]
    x = MX
    for val, label in kpis:
        text(s, val, x, 5.66, 3.85, 0.44, size=19, bold=True, color=NAVY)
        text(s, label, x, 6.12, 3.85, 0.30, size=10.2, color=MUTED)
        x += 3.95

    text(s, "checkpoint 按独立验证切片的 Data BER 选择，而不是 Pilot 上的 BCE。",
         MX, 6.54, CW, 0.30, size=10.5, color=MUTED)
    speaker(s, "这是离线训练的结构。接收帧和只在 Adapt Pilot 位置非零的已知符号一起送入网络，"
               "物理条件分支用 acquisition CIR 做 warm start，主干是 4 层迭代展开。"
               "损失是整帧的等权交叉熵，但选模型看的是独立验证切片上的 Data BER。")
    return s


def slide_traditional(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 公平对照",
             "不是靠弱基线取胜：同波形同 Pilot 下传统接收机差 2–12 倍")
    pic_fit(s, FIGDIR / "v2" / "fig_traditional_fair.png", MX, 1.58, 8.55, 4.15)

    x = 9.35
    items = [
        ("公平性", "同 episode、同接收波形、同 Pilot 预算、同 acquisition CIR"),
        ("传统方法", "不读 Reward/Data 标签，不使用神经网络或强化学习"),
        ("配对差", "每个 SNR 48 对，传统减 Bayes 全部为正"),
        ("边界", "3 seed × 16 帧开发规模，不是确认级证据"),
    ]
    y = 1.75
    for h, body in items:
        rect(s, x, y + 0.05, 0.045, 0.26, fill=BAYES)
        text(s, h, x + 0.18, y, 3.30, 0.30, size=11.5, bold=True, color=NAVY)
        text(s, body, x + 0.18, y + 0.32, 3.30, 0.85, size=9.8, color=MUTED)
        y += 1.06

    text(s, "倍数 = 较强的传统方法 BER ÷ Bayes BER。SC-FDE-MMSE 未做 CFO/相位补偿，只作弱对照。",
         MX, 6.24, CW, 0.30, size=9.8, color=MUTED)
    speaker(s, "这一页回答一个可能的质疑：是不是传统基线被削弱了？"
               "公平对照里传统接收机用的是同一批波形、同一 Pilot 预算、同一 acquisition CIR，"
               "而且不读任何标签。结果 Bayes 仍然低 2 到 12 倍，高信噪比差距更大。"
               "注意这是 3×16 的开发规模，不是确认级证据。")
    return s



def slide_cir(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 问题定位",
             "离线网络已饱和，剩余误差主要来自 CIR 估计",
             "离线饱和诊断：换更大模型、续训更多步都无改善；换成真实 CIR 则明显改善")
    pic_fit(s, FIGDIR / "fig_oracle_cir.png", MX, 1.85, 7.55, 4.40)

    x = 8.45
    text(s, "离线侧已到平台", x, 1.95, 4.2, 0.30, size=12.5, bold=True, color=INK)
    text(s, "续训 32/64 步、d_model 64→96、iterations 4→6，BER 基本不变",
         x, 2.28, 4.2, 0.75, size=10.5, color=MUTED)

    text(s, "瓶颈在物理状态", x, 3.30, 4.2, 0.30, size=12.5, bold=True, color=BAYES)
    text(s, "同一网络、换真实 CIR，5 dB 从 5.28% 降到 4.20%",
         x, 3.63, 4.2, 0.75, size=10.5, color=MUTED)

    text(s, "于是转向在线跟踪", x, 4.65, 4.2, 0.30, size=12.5, bold=True, color=INK)
    text(s, "不动神经权重，只用合法 Pilot 更新物理状态",
         x, 4.98, 4.2, 0.75, size=10.5, color=MUTED)
    speaker(s, "先排除一种可能：是不是网络不够大、训练不够久？"
               "实验表明都已饱和。但把估计 CIR 换成真实 CIR，BER 明显下降。"
               "所以瓶颈在物理状态估计，而不是神经网络容量。")
    return s


def slide_arch(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 方法",
             "在线只读合法 Pilot：Data 标签不参与任何更新与决策")
    pic_fit(s, FIGDIR / "fig_rx_arch.png", MX, 1.55, CW, 4.45)
    text(s, "Data 段波形参与均衡与无标签更新，Data 标签只用于事后评分　→　Frozen 与 Bayes 严格配对",
         MX, 6.20, CW, 0.30, size=11.5, color=MUTED)
    speaker(s, "接收机结构：Adapt Pilot 只用于更新物理状态，Data 段只做事后评分，"
               "Reward Pilot 只在动作之后验收。Data 标签这条路径是完全断开的，"
               "这是整个方法可部署性的前提。")
    return s


def slide_loop(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 在线闭环",
             "在线闭环：状态条件化均衡，判决回来修正状态")
    pic_fit(s, FIGDIR / "fig_online_loop.png", MX, 1.52, CW, 4.42)
    text(s, "四类物理量：CIR（116 抽头）· CFO · 慢相位 · soft-tail（上一帧末尾 116 个软判决）",
         MX, 6.10, CW, 0.30, size=10.5, color=MUTED)
    text(s, "Data 后验用模型软判决、不读标签；网络权重全帧冻结，每帧只更新这四类物理量。",
         MX, 6.46, CW, 0.30, size=10.5, color=MUTED)
    speaker(s, "这是在线阶段的闭环。每一帧先用当前的物理状态去条件化冻结网络，得到软判决；"
               "判决出来之后，再用 Adapt Pilot 的已知符号和这批软判决，把状态更新一次，传给下一帧。"
               "Soft 判决的噪声项里带符号方差，所以低信噪比下判错的符号会自动降权。"
               "四类状态里 CIR 是主要项，CFO 和慢相位决定整帧的相位，"
               "soft-tail 处理跨越帧边界的 116 个符号记忆。")
    return s


def slide_scale(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 开发筛查",
             "三种开发规模上，配对增益的方向始终一致")
    pic_fit(s, FIGDIR / "v2" / "fig_scale.png", 1.55, 1.50, 10.2, 4.75)
    text(s, "5 seed × 32 帧　→　10 seed × 64 帧　→　10 seed × 128 帧",
         MX, 6.30, CW, 0.30, size=11.5, color=MUTED)
    speaker(s, "在正式确认之前，先在开发 seed 上做了三次规模递增的筛查。"
               "每个 SNR 的增益方向都一致，样本量增加时区间总体收窄。")
    return s


def slide_growth(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 时序行为",
             "增益在会话内累积：5/10/15 dB 后段仍在上升，0 dB 峰值后回落")
    pic_fit(s, FIGDIR / "v2" / "fig_growth.png", MX, 1.52, 9.45, 4.75)
    x = 10.30
    text(s, "物理状态在成长", x, 1.90, 2.60, 0.30, size=12.5, bold=True, color=BAYES)
    text(s, "不是网络权重在变", x, 2.22, 2.60, 0.32, size=10.5, color=MUTED)
    text(s, "是 CIR 后验在收敛", x, 2.56, 2.60, 0.32, size=10.5, color=MUTED)
    text(s, "0 dB 的回落", x, 3.55, 2.60, 0.30, size=12.5, bold=True, color=INK)
    text(s, "低 SNR 下 soft 判决误差大，后验波动也更明显", x, 3.87, 2.60, 1.05,
         size=10.5, color=MUTED)
    speaker(s, "把 128 帧切成 8 个窗口看趋势。5/10/15 dB 后段还在涨，"
               "0 dB 在 96 到 112 帧附近见顶然后回落。这个现象符合后验收敛后又受噪声扰动的机制。")
    return s


def slide_confirm(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 独立确认 v1",
             "四个 SNR 全部显著，单侧 95% 下界均为正",
             "独立 seed 100001–100010，10 session × 128 帧；配置与 checkpoint 运行前冻结")
    pic_fit(s, FIGDIR / "v2" / "fig_ber_forest.png", MX, 1.80, CW, 4.35)
    text(s, "H1：四 SNR 等权平均 +0.01177，下界 +0.01019，p = 1.26×10⁻⁷",
         MX, 6.28, CW, 0.30, size=11.5, color=INK)
    speaker(s, "这是核心结果。独立 seed、128 帧长会话、配置运行前冻结。"
               "四个 SNR 的配对增益全部为正，置信下界也都大于零，"
               "Holm 校正后 p 值最大只有 1e-3。")
    return s


def slide_seed(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 独立确认 v1",
             "10 个独立 session，配对增益无一个为负")
    pic_fit(s, FIGDIR / "v2" / "fig_seed.png", MX, 1.55, CW, 3.55)

    rows = [
        ("H1", "四 SNR 等权平均增益 > 0", "+0.01177　下界 +0.01019", "通过"),
        ("H2", "每个 SNR 下界 > 0 且无负向 seed", "0/5/10/15 dB 全部满足", "通过"),
        ("H3", "增益随帧增长", "5/10/15 dB 通过；0 dB 早期收敛", "部分"),
    ]
    y = 5.28
    for name, desc, val, verdict in rows:
        vc = GOOD if verdict == "通过" else BAYES
        text(s, name, MX, y, 0.55, 0.30, size=11, bold=True, color=NAVY)
        text(s, desc, MX + 0.65, y, 4.4, 0.30, size=10.5, color=INK)
        text(s, val, MX + 5.2, y, 4.6, 0.30, size=10.5, color=MUTED)
        text(s, verdict, SW - MX - 1.1, y, 1.1, 0.30, size=10.5, bold=True,
             color=vc, align=PP_ALIGN.RIGHT)
        y += 0.42
        if name != "H3":
            hline(s, MX, y - 0.06, SW - MX, color="#EDEAE4")
    speaker(s, "逐 session 看，10 个 session 的配对增益全部为正，没有一个是负的。"
               "H1 和 H2 通过。H3 的增长假设在 5/10/15 dB 成立；0 dB 呈早期收敛、后段饱和，按预注册判据不计为增长，"
               "这一点我们如实报告。")
    return s


def slide_boundary(prs, idx, total):
    s = page(prs, idx, total, "研究点二 · 边界",
             "物理增益已在四 SNR 确认，边界在 0 dB 饱和与成本度量")

    # 左：NMSE
    pic_fit(s, FIGDIR / "v2" / "fig_nmse.png", MX, 1.58, 7.45, 4.55)
    text(s, "同一波形下 CIR 估计误差下降 67%–75%", MX, 6.22, 7.45, 0.30,
         size=10.5, color=MUTED)

    # 右：两栏清单
    x = 8.55
    rect(s, x, 1.62, 0.05, 0.34, fill=GOOD)
    text(s, "已经确认", x + 0.18, 1.62, 4.3, 0.34, size=13, bold=True, color=GOOD)
    ok = ["离线神经均衡显著优于传统基线", "物理 Bayes CIR 在线适配增益", "四 SNR 独立确认，H1/H2 通过",
          "增益随会话帧数累积", "在线不使用任何 Data 标签"]
    y = 2.08
    for t in ok:
        text(s, "·　" + t, x + 0.18, y, 4.5, 0.30, size=10.8, color=INK)
        y += 0.36

    rect(s, x, 3.96, 0.05, 0.34, fill=ALERT)
    text(s, "需要说明的边界", x + 0.18, 3.96, 4.3, 0.34, size=13, bold=True, color=ALERT)
    no = ["0 dB 早期收敛/后段饱和，不计为增长", "确认集只运行一次，不得重测",
          "逐帧延迟与显存尚未插桩"]
    y = 4.42
    for t in no:
        text(s, "·　" + t, x + 0.18, y, 4.5, 0.30, size=10.8, color=INK)
        y += 0.36

    text(s, "物理增益、离线训练可复现性分别报告。", x + 0.18, 5.55, 4.5, 0.30,
         size=9.5, color=MUTED)
    speaker(s, "物理状态适配的增益经过独立确认：四个 SNR 全部显著，10 个 session 无一为负。"
               "需要说明的边界有三点：0 dB 的增益早期就饱和，增长假设不通过；"
               "确认集只跑了一次；逐帧延迟和显存还没有插桩。")
    return s


def slide_plan(prs, idx, total):
    s = page(prs, idx, total, "后续工作",
             "下一步：扩展会话与漂移场景，完成论文主体")
    pic_fit(s, FIGDIR / "fig_plan.png", MX, 1.58, CW, 3.95)
    text(s, "保持同一信道、Pilot 协议与统计判据，只扩展会话长度与漂移条件。",
         MX, 5.12, CW, 0.30, size=11.5, color=INK)
    text(s, "确认判据在运行前冻结，不以开发集调参替代确认。",
         MX, 5.50, CW, 0.30, size=10.5, color=MUTED)
    speaker(s, "下一步保持信道、Pilot 和统计判据不变，只把会话拉长、加入漂移条件，"
               "检验增益在更长会话和信道变化下是否保持；同时补齐延迟与显存的测量。")
    return s


def slide_summary(prs, idx, total):
    s = page(prs, idx, total, "总结", "两个研究点：一个已完成，一个从零走到可确认")

    cards = [
        ("研究点一", "已完成", "统一积分模型\n长记忆多径建模\n拟录用论文", BLUE),
        ("研究点二", "已确认", "物理 Bayes CIR\n四 SNR 独立确认\n增益随帧累积", BAYES),
        ("下一步", "推进中", "会话扩展与漂移\n冻结确认判据\n论文整合", TEAL),
    ]
    x = MX
    for tag, status, body, c in cards:
        rect(s, x, 1.70, 3.75, 2.95, fill=PANEL)
        rect(s, x, 1.70, 3.75, 0.06, fill=c)
        text(s, tag, x + 0.32, 1.96, 3.1, 0.28, size=10.5, color=MUTED)
        text(s, status, x + 0.32, 2.24, 3.1, 0.60, size=26, bold=True, color=c)
        text(s, body, x + 0.32, 3.02, 3.15, 1.45, size=11.5, color=INK, line=1.55)
        x += 4.04

    deco = s.shapes.add_picture(str(BRAND / "main_building.png"),
                                Inches(9.15), Inches(5.42), width=Inches(3.30))
    set_alpha(deco, 20)

    hline(s, MX, 5.35, SW - MX, color=LINE_CLR)
    text(s, "谢谢各位老师，请批评指正。", MX, 5.62, CW, 0.45, size=16, color=INK)
    text(s, "2026 年 11 月　·　李诚谦", MX, 6.24, CW, 0.30, size=10, color=MUTED)
    speaker(s, "总结一下：第一研究点完成，形成拟录用论文；第二研究点从零开始，"
               "现在物理层面的在线适配已经通过独立确认；下一步扩展会话并完成论文。谢谢。")
    return s


# ---------------------------------------------------------------- 组装


def main():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    builders = [
        cover, slide_route, slide_model, slide_rcs, slide_delay, slide_baseline,
        slide_offline, slide_offline_arch, slide_traditional, slide_cir, slide_arch, slide_loop, slide_scale,
        slide_growth, slide_confirm, slide_seed, slide_boundary, slide_plan,
        slide_summary,
    ]
    total = len(builders)
    for i, fn in enumerate(builders, 1):
        fn(prs, i, total)
    print(f"{total} 页")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"已生成：{OUT}（{len(prs.slides)} 页）")


if __name__ == "__main__":
    main()

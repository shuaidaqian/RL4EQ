"""按开题《内容目标.docx》格式生成「研究内容」docx，并插入示意图。

示意图用于说明方法与场景；数据/结论图留在「工作成果」栏。
"""

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
SRC = Path("D:/Research/开题/内容目标.docx")
DST = HERE / "研究内容.docx"

# 插入点：(定位关键词, 图片, 图题)
FIGURES = [
    ("回波由准镜面反射与漫散射两部分构成",
     "figures/paper/fig1.png", "宽带 EME 统一积分模型反射场景示意"),
    ("本课题的研究内容分为两个研究点",
     "figures/fig_route.png", "本课题研究主线：从信道建模到在线自适应接收机"),
    ("展开结构把",
     "figures/fig_offline_arch.png", "Pilot 条件化迭代展开均衡器的结构与训练流程"),
    ("后验协方差跨帧持久并叠加过程噪声",
     "figures/fig_online_loop.png", "在线适配闭环：状态条件化、判决与状态更新"),
]

NUM = {"一、选题背景": "1. 选题背景", "二、研究内容": "2. 研究内容",
       "三、关键技术": "3. 关键技术", "四、工作计划": "4. 工作计划",
       "五、工作进度及目标": "5. 工作进度及目标"}


def add_picture(doc, path, caption, seq):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Emu(0)
    p.paragraph_format.left_indent = Emu(0)
    p.paragraph_format.space_before = Emu(99060)
    p.add_run().add_picture(str(path), width=Inches(5.1))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.first_line_indent = Emu(0)
    c.paragraph_format.left_indent = Emu(0)
    r = c.add_run(f"图 {seq}　{caption}")
    r.font.size = Pt(9)
    r.bold = False


def main():
    shutil.copy(SRC, DST)
    doc = Document(str(DST))
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    txt = (HERE / "中期报告_研究内容栏_纯文本.txt").read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in txt.split("\n") if b.strip()]

    seq = 0
    for b in blocks:
        if b in NUM:
            p = doc.add_paragraph(style="List Paragraph")
            r = p.add_run(NUM[b]); r.bold = True
            p.paragraph_format.first_line_indent = Emu(267970)
        else:
            p = doc.add_paragraph(); p.add_run(b)
            p.paragraph_format.first_line_indent = Emu(279400)
        pf = p.paragraph_format
        pf.left_indent = Emu(279400); pf.line_spacing = 1.2; pf.space_before = Emu(99060)

        for anchor, img, cap in FIGURES:
            if anchor in b and (HERE / img).exists():
                seq += 1
                add_picture(doc, HERE / img, cap, seq)

    doc.save(str(DST))
    print(f"已生成 {DST}，插入 {seq} 张示意图")


if __name__ == "__main__":
    main()

"""生成「工作成果」docx：格式同开题模板，正文中插入结果数据图。"""

import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Emu, Inches, Pt
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = Path("D:/Research/开题/内容目标.docx")
TXT = HERE / "工作成果_纯文本.txt"
OUT = HERE / "工作成果.docx"

FIG_PAT = re.compile(r"^\[FIG:([^|]+)\|(.+)\]$")


def add_figure(doc, img: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.first_line_indent = Emu(0); pf.left_indent = Emu(0); pf.space_before = Emu(99060)
    w, h = Image.open(str(img)).size
    width = 4.3 if (w / h) < 1.6 else 5.4
    p.add_run().add_picture(str(img), width=Inches(width))

    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cf = c.paragraph_format
    cf.first_line_indent = Emu(0); cf.left_indent = Emu(0)
    r = c.add_run(caption)
    r.font.size = Pt(9); r.bold = False


def main():
    shutil.copy(SRC, OUT)
    doc = Document(str(OUT))
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    blocks = [b.strip() for b in TXT.read_text(encoding="utf-8").split("\n") if b.strip()]
    figs = 0
    for b in blocks:
        m = FIG_PAT.match(b)
        if m:
            img = HERE / "figures" / m.group(1)
            if img.exists():
                add_figure(doc, img, m.group(2))
                figs += 1
            else:
                print("缺图:", img)
            continue

        is_heading = b.startswith(("一、", "二、", "三、", "四、", "五、"))
        if is_heading:
            p = doc.add_paragraph(style="List Paragraph")
            r = p.add_run(b); r.bold = True
            p.paragraph_format.first_line_indent = Emu(267970)
        else:
            p = doc.add_paragraph(); p.add_run(b)
            p.paragraph_format.first_line_indent = Emu(279400)
        pf = p.paragraph_format
        pf.left_indent = Emu(279400); pf.line_spacing = 1.2; pf.space_before = Emu(99060)

    doc.save(str(OUT))
    print(f"已生成 {OUT}，插入 {figs} 张图")


if __name__ == "__main__":
    main()

"""把 midterm_report_v2.md 生成学校中期报告格式的 docx。"""

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
MD = HERE / "midterm_report_v2.md"
OUT = HERE / "中期报告_李诚谦_20261106.docx"

NAVY = "1B3A57"
TEAL = "2E6F8E"

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.8)
sec.bottom_margin = Inches(0.75)
sec.left_margin = Inches(0.9)
sec.right_margin = Inches(0.9)

normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal.font.size = Pt(11)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

for name, size, color in [
    ("Title", 20, NAVY), ("Heading 1", 15, NAVY), ("Heading 2", 12.5, TEAL),
]:
    st = doc.styles[name]
    st.font.name = "黑体"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    st.paragraph_format.space_before = Pt(10)
    st.paragraph_format.space_after = Pt(6)


def add_inline(p, s, size=None):
    """处理 **加粗** 与行内代码。"""
    s = s.replace("`", "")
    for chunk in re.split(r"(\*\*.*?\*\*)", s):
        if not chunk:
            continue
        run = p.add_run(chunk[2:-2] if chunk.startswith("**") and chunk.endswith("**") else chunk)
        if chunk.startswith("**") and chunk.endswith("**"):
            run.bold = True
        if size:
            run.font.size = Pt(size)


lines = MD.read_text(encoding="utf-8").splitlines()
first_title = True
i = 0
while i < len(lines):
    s = lines[i].strip()
    if not s:
        i += 1
        continue

    if s.startswith("# "):
        p = doc.add_paragraph(style="Title")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(s[2:])
        first_title = False
        i += 1
    elif s.startswith("## "):
        doc.add_heading(s[3:], level=1)
        i += 1
    elif s.startswith("### "):
        doc.add_heading(s[4:], level=2)
        i += 1
    elif s.startswith("| "):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
            i += 1
        if rows:
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for r in rows:
                cs = table.add_row().cells
                for cell, val in zip(cs, r):
                    cell.text = val.replace("`", "")
                    for par in cell.paragraphs:
                        for run in par.runs:
                            run.font.size = Pt(9)
            doc.add_paragraph()
    elif re.match(r"^\d+\.\s", s):
        # 参考文献条目：悬挂缩进，不加首行缩进
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.32)
        p.paragraph_format.first_line_indent = Inches(-0.32)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.2
        add_inline(p, s, size=9.5)
        i += 1
    elif s.startswith("**") and s.count("**") >= 2:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Inches(0.28)
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_after = Pt(5)
        add_inline(p, s)
        i += 1
    else:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Inches(0.28)
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_after = Pt(5)
        add_inline(p, s)
        i += 1

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run("北京邮电大学信息与通信工程学院　|　硕士学位论文中期报告")
run.font.size = Pt(9)
run.font.color.rgb = RGBColor.from_string("6B7A89")

doc.save(OUT)
print(f"已生成 {OUT}")

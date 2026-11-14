"""按开题《内容目标.docx》格式，把分节纯文本转成 docx。

用法：python build_section_docx.py <文本文件> <输出docx>
"""

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.shared import Emu

HERE = Path(__file__).resolve().parent
SRC = Path("D:/Research/开题/内容目标.docx")


def build(text_file: Path, out_file: Path):
    shutil.copy(SRC, out_file)
    doc = Document(str(out_file))
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    blocks = [b.strip() for b in text_file.read_text(encoding="utf-8").split("\n") if b.strip()]
    for b in blocks:
        is_heading = b.startswith(("一、", "二、", "三、", "四、", "五、", "六、"))
        if is_heading:
            p = doc.add_paragraph(style="List Paragraph")
            r = p.add_run(b)
            r.bold = True
            p.paragraph_format.first_line_indent = Emu(267970)
        else:
            p = doc.add_paragraph()
            p.add_run(b)
            p.paragraph_format.first_line_indent = Emu(279400)
        pf = p.paragraph_format
        pf.left_indent = Emu(279400)
        pf.line_spacing = 1.2
        pf.space_before = Emu(99060)

    doc.save(str(out_file))
    print(f"已生成 {out_file}")


if __name__ == "__main__":
    build(HERE / sys.argv[1], HERE / sys.argv[2])

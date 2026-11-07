"""PPTX 结构自检：越界、空页、备注缺失、图片分辨率。"""

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Emu

HERE = Path(__file__).resolve().parent
PPTX = HERE / "中期答辩_李诚谦_v4.pptx"

prs = Presentation()
prs = Presentation(str(PPTX))
SW, SH = prs.slide_width, prs.slide_height
tol = Emu(9144)  # 0.01 英寸

problems = []
for i, slide in enumerate(prs.slides, 1):
    pics = 0
    for sh in slide.shapes:
        if sh.left is None:
            continue
        r, b = sh.left + (sh.width or 0), sh.top + (sh.height or 0)
        if r > SW + tol or b > SH + tol or sh.left < -tol or sh.top < -tol:
            problems.append(
                f"p{i} 越界: {sh.shape_type} "
                f"L{sh.left/914400:.2f} T{sh.top/914400:.2f} "
                f"R{r/914400:.2f} B{b/914400:.2f}"
            )
        if sh.shape_type == 13:
            pics += 1
            try:
                im = Image.open(sh.image.blob and __import__("io").BytesIO(sh.image.blob))
                w_in = sh.width / 914400
                dpi = im.size[0] / w_in
                if dpi < 90:
                    problems.append(f"p{i} 图片分辨率偏低: {dpi:.0f} dpi, {w_in:.2f}in")
            except Exception:
                pass
    if not slide.has_notes_slide or not slide.notes_slide.notes_text_frame.text.strip():
        problems.append(f"p{i} 缺少演讲者备注")
    if pics == 0 and i not in (1, len(prs.slides)):
        problems.append(f"p{i} 无图")

print(f"共 {len(prs.slides)} 页")
if problems:
    print("发现问题：")
    for p in problems:
        print("  -", p)
else:
    print("结构检查通过：无越界、无缺备注、图片分辨率达标。")

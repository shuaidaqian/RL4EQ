"""用本机 PowerPoint 把 PPTX 逐页导出为 PNG，供视觉检查。"""

import shutil
from pathlib import Path

import win32com.client

HERE = Path(__file__).resolve().parent
PPTX = HERE / "中期答辩_李诚谦_v4.pptx"
OUT = HERE / "render"

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()

app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(str(PPTX), WithWindow=False)
try:
    n = pres.Slides.Count
    for i in range(1, n + 1):
        pres.Slides(i).Export(str(OUT / f"slide{i:02d}.png"), "PNG", 1600, 900)
    print(f"导出 {n} 页 → {OUT}")
finally:
    pres.Close()
    app.Quit()

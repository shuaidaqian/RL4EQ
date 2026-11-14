"""导出 PDF 版，便于快速翻阅。"""

from pathlib import Path

import win32com.client

HERE = Path(__file__).resolve().parent
PPTX = HERE / "中期答辩_李诚谦_v4.pptx"
PDF = HERE / "中期答辩_李诚谦_v4.pdf"

app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(str(PPTX), WithWindow=False)
try:
    pres.SaveAs(str(PDF), 32)  # ppSaveAsPDF
    print(f"已导出 {PDF}")
finally:
    pres.Close()
    app.Quit()

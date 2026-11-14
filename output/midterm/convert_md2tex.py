"""把 paper_draft_v1.md 转成 paper/sections/*.tex。"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "paper_draft_v1.md"
OUT = HERE.parents[1] / "paper" / "sections"

SECTION_FILES = {
    "1": "01_intro.tex", "2": "02_model.tex", "3": "03_offline.tex",
    "4": "04_online.tex", "5": "05_setup.tex", "6": "06_results.tex",
    "7": "07_discussion.tex", "8": "08_conclusion.tex",
}


def esc(s):
    """转义 LaTeX 特殊字符，但保留 $...$ 公式段。"""
    parts = re.split(r"(\$[^$]*\$)", s)
    out = []
    for k, seg in enumerate(parts):
        if k % 2 == 1:
            out.append(seg)
            continue
        seg = seg.replace("\\", r"\textbackslash{}")
        for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                     ("_", r"\_"), ("^", r"\^{}"), ("~", r"\textasciitilde{}")]:
            seg = seg.replace(a, b)
        out.append(seg)
    s = "".join(out)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    return s


def flush_table(rows, caption, buf):
    ncol = len(rows[0])
    align = "l" + "r" * (ncol - 1)
    buf.append("\\begin{table}[htbp]")
    buf.append("\\centering")
    if caption:
        buf.append(f"\\caption{{{esc(caption)}}}")
    buf.append(f"\\begin{{tabular}}{{{align}}}")
    buf.append("\\toprule")
    buf.append(" & ".join(esc(c) for c in rows[0]) + r" \\")
    buf.append("\\midrule")
    for r in rows[1:]:
        buf.append(" & ".join(esc(c) for c in r) + r" \\")
    buf.append("\\bottomrule")
    buf.append("\\end{tabular}")
    buf.append("\\end{table}")
    buf.append("")


def flush_figure(path, caption, buf):
    buf.append("\\begin{figure}[htbp]")
    buf.append("\\centering")
    buf.append(f"\\includegraphics[width=0.95\\linewidth]{{{path}}}")
    if caption:
        buf.append(f"\\caption{{{esc(caption)}}}")
    buf.append("\\end{figure}")
    buf.append("")


def main():
    lines = SRC.read_text(encoding="utf-8").split("\n")
    OUT.mkdir(parents=True, exist_ok=True)

    cur_file, buf = None, []
    pending_table = None      # (rows,)
    pending_fig = None        # path
    abstract = []

    def save():
        if cur_file is not None:
            (OUT / cur_file).write_text("\n".join(buf).strip() + "\n", encoding="utf-8")

    i = 0
    while i < len(lines):
        s = lines[i].rstrip()
        t = s.strip()

        # 标题
        if t.startswith("# ") and not t.startswith("## "):
            i += 1
            continue
        if t.startswith("## "):
            head = t[3:].strip()
            if head == "摘要":
                cur_file = None
                buf = abstract
                i += 1
                continue
            if head.startswith("参考文献"):
                save()
                cur_file, buf = None, []
                i += 1
                continue
            m = re.match(r"(\d+)\s+(.*)", head)
            if m and m.group(1) in SECTION_FILES:
                save()
                cur_file, buf = SECTION_FILES[m.group(1)], []
                buf.append(f"\\section{{{esc(m.group(2))}}}")
                buf.append("")
                i += 1
                continue
        if t.startswith("### "):
            head = re.sub(r"^\d+(\.\d+)*\s*", "", t[4:].strip())
            buf.append(f"\\subsection{{{esc(head)}}}")
            buf.append("")
            i += 1
            continue

        # 公式块
        if t == "$$":
            j = i + 1
            body = []
            while j < len(lines) and lines[j].strip() != "$$":
                body.append(lines[j])
                j += 1
            buf.append("\\begin{equation*}")
            buf.extend(body)
            buf.append("\\end{equation*}")
            buf.append("")
            i = j + 1
            continue

        # 图
        m = re.match(r"!\[\]\((.+?)\)", t)
        if m:
            pending_fig = m.group(1)
            i += 1
            continue

        # 表
        if t.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            pending_table = rows
            continue

        # 图注 / 表注（兼容有无 ** 两种写法）
        m = re.match(r"^\*{0,2}(图|表)\s*\d+\*{0,2}\s*(.+)", t)
        if m and (pending_fig or pending_table):
            cap = m.group(2).strip()
            if pending_fig:
                flush_figure(pending_fig, cap, buf)
                pending_fig = None
            else:
                flush_table(pending_table, cap, buf)
                pending_table = None
            i += 1
            continue

        # 空行
        if not t:
            i += 1
            continue

        # 普通段落
        para = [t]
        # 合并被硬换行拆开的同一段
        while i + 1 < len(lines) and lines[i + 1].strip() and \
                not re.match(r"^(#{1,3} |\$\$|\||!\[|\*\*(图|表))", lines[i + 1].strip()):
            i += 1
            para.append(lines[i].strip())
        buf.append(esc("".join(para)))
        buf.append("")
        i += 1

    save()
    (OUT / "00_abstract.tex").write_text(
        "\\begin{abstract}\n" + "\n".join(esc(x) if x.strip() else "" for x in abstract).strip()
        + "\n\\end{abstract}\n", encoding="utf-8")
    print("转换完成，输出到", OUT)


if __name__ == "__main__":
    main()

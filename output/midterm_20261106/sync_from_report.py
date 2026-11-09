"""把 LwMidReport.docx 中用户删减后的各栏文本，同步到对应的栏目 docx。

只修改栏目 docx；不触碰 LwMidReport.docx。
"""

import difflib
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
REPORT = HERE / "LwMidReport.docx"

PAIRS = {
    "二、工作进展情况": "进展情况.docx",
    "三、工作成果": "工作成果.docx",
    "四、计划及进度安排": "计划及进度安排.docx",
    "五、问题及整改方案": "问题及整改方案.docx",
    "六、参考文献": "参考文献.docx",
}


def set_text(p, text):
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


def extract_report():
    d = Document(str(REPORT))
    out = {}
    for t in d.tables:
        for r in t.rows:
            cells = [c.text.strip() for c in r.cells]
            if not cells:
                continue
            text = max(cells, key=len)
            first = text.split("\n")[0].strip()
            if first.startswith(("二、", "三、", "四、", "五、", "六、")):
                out[first] = [l.strip() for l in text.split("\n") if l.strip()][1:]
    return out


def sync(docx_path: Path, new_paras):
    doc = Document(str(docx_path))
    paras = [p for p in doc.paragraphs if p.text.strip()]
    old = [p.text.strip() for p in paras]

    sm = difflib.SequenceMatcher(None, old, new_paras, autojunk=False)
    changed = {"replace": 0, "delete": 0, "insert": 0}

    for tag, i1, i2, j1, j2 in reversed(sm.get_opcodes()):
        if tag == "equal":
            continue
        if tag == "replace":
            n_old, n_new = i2 - i1, j2 - j1
            for k in range(min(n_old, n_new)):
                set_text(paras[i1 + k], new_paras[j1 + k])
            if n_old > n_new:
                for k in range(i1 + n_new, i2):
                    paras[k]._element.getparent().remove(paras[k]._element)
            changed["replace"] += min(n_old, n_new)
        elif tag == "delete":
            for k in range(i1, i2):
                paras[k]._element.getparent().remove(paras[k]._element)
            changed["delete"] += i2 - i1
        elif tag == "insert":
            changed["insert"] += j2 - j1

    doc.save(str(docx_path))
    return changed


def main():
    lw = extract_report()
    for sec, fname in PAIRS.items():
        f = HERE / fname
        if not f.exists() or sec not in lw:
            print(f"跳过 {sec}")
            continue
        ch = sync(f, lw[sec])
        print(f"{fname}: 替换 {ch['replace']}，删除 {ch['delete']}，待插入 {ch['insert']}")


if __name__ == "__main__":
    main()

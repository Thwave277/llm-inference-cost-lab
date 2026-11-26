# -*- coding: utf-8 -*-
"""REPORT.md -> docx. Single source of truth: the markdown file."""
import io
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Microsoft YaHei"
CODE_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x3B, 0x57)
GRAY = RGBColor(0x55, 0x55, 0x55)


def set_east_asia(run, name):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), name)
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)


TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(p, text, size, color=None, bold=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        is_bold = bold
        is_code = False
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            part = part[2:-2]
            is_bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            part = part[1:-1]
            is_code = True
        r = p.add_run(part)
        r.font.size = Pt(size - 0.5 if is_code else size)
        r.bold = is_bold
        set_east_asia(r, CODE_FONT if is_code else FONT)
        if color is not None:
            r.font.color.rgb = color


def new_para(doc, space_after=6, indent=0.0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.32
    if indent:
        pf.left_indent = Cm(indent)
    return p


def code_block(doc, lines):
    p = new_para(doc, space_after=8, indent=0.35)
    p.paragraph_format.line_spacing = 1.15
    for i, line in enumerate(lines):
        if i:
            p.add_run().add_break()
        r = p.add_run(line)
        r.font.size = Pt(8.5)
        set_east_asia(r, CODE_FONT)
        r.font.color.rgb = GRAY


def make_table(doc, rows, header_size=9):
    cols = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=cols)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci in range(cols):
            text = row[ci] if ci < len(row) else ""
            cell = cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            add_inline(p, text, header_size, bold=(ri == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def is_sep(cells):
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip() != "") and any(cells)


def build(md_path, out_path):
    text = io.open(md_path, encoding="utf-8").read()
    lines = text.split("\n")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(1.9)

    i, n = 0, len(lines)
    in_code = False
    buf = []

    while i < n:
        line = lines[i]
        s = line.strip()

        if s.startswith("```"):
            if not in_code:
                in_code, buf = True, []
            else:
                code_block(doc, buf)
                in_code = False
            i += 1
            continue
        if in_code:
            buf.append(line)
            i += 1
            continue

        if not s:
            i += 1
            continue

        if s == "---":
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            body = m.group(2)
            if level == 1:
                p = new_para(doc, space_after=2)
                add_inline(p, body, 19, color=ACCENT, bold=True)
            else:
                size = {2: 14, 3: 11.5, 4: 10.5}[level]
                p = new_para(doc, space_after=5)
                p.paragraph_format.space_before = Pt(13 if level == 2 else 9)
                p.paragraph_format.keep_with_next = True
                add_inline(p, body, size, color=ACCENT, bold=True)
            i += 1
            continue

        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not is_sep(cells):
                    rows.append(cells)
                i += 1
            if rows:
                make_table(doc, rows)
            continue

        if re.match(r"^[-*]\s+", s):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.3
            add_inline(p, re.sub(r"^[-*]\s+", "", s), 10.5)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", s):
            p = new_para(doc, space_after=4, indent=0.0)
            add_inline(p, s, 10.5)
            i += 1
            continue

        p = new_para(doc)
        add_inline(p, s, 10.5)
        i += 1

    doc.save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])

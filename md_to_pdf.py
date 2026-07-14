#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Markdown 商业计划书转换为 PDF。
支持中文、标题、段落、加粗、有序/无序列表、表格、引用块、分隔线、代码块。
"""

import re
import sys
from pathlib import Path

from fpdf import FPDF


class MarkdownPDF(FPDF):
    def __init__(self, font_path: str, font_bold_path: str):
        super().__init__()
        self.add_font("CN", "", font_path, uni=True)
        self.add_font("CN", "B", font_bold_path, uni=True)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("CN", "", 11)
        self.base_line_height = 6.2
        self.content_x = self.l_margin + 6
        self.content_width = self.w - self.l_margin - self.r_margin - 12

    def render_markdown(self, md_text: str):
        lines = md_text.splitlines()
        i = 0
        in_code = False
        code_lines = []
        blockquote_buffer = []

        def flush_blockquote():
            if blockquote_buffer:
                self.render_blockquote("\n".join(blockquote_buffer))
                blockquote_buffer.clear()

        while i < len(lines):
            raw_line = lines[i]
            stripped = raw_line.strip()

            # 代码块
            if stripped.startswith("```"):
                if in_code:
                    self.render_code_block("\n".join(code_lines))
                    code_lines = []
                    in_code = False
                else:
                    flush_blockquote()
                    in_code = True
                i += 1
                continue

            if in_code:
                code_lines.append(raw_line)
                i += 1
                continue

            # 空行
            if stripped == "":
                flush_blockquote()
                i += 1
                continue

            # 引用块
            if stripped.startswith("> "):
                blockquote_buffer.append(stripped[2:])
                i += 1
                continue
            flush_blockquote()

            # 分隔线
            if re.fullmatch(r"[-*]{3,}", stripped):
                self.render_hr()
                i += 1
                continue

            # 表格
            if "|" in raw_line and i + 1 < len(lines) and re.search(r"\|[-:\s]+", lines[i + 1]):
                table_rows = []
                while i < len(lines) and "|" in lines[i]:
                    table_rows.append(lines[i])
                    i += 1
                self.render_table(table_rows)
                continue

            # 标题
            header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2).strip()
                self.render_header(level, text)
                i += 1
                continue

            # 有序列表
            ol_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if ol_match:
                self.render_list_item(ol_match.group(2), ol_match.group(1) + ".")
                i += 1
                continue

            # 无序列表
            ul_match = re.match(r"^[-*]\s+(.*)$", stripped)
            if ul_match:
                self.render_list_item(ul_match.group(1), "•")
                i += 1
                continue

            # 普通段落
            self.render_paragraph(stripped)
            i += 1

        flush_blockquote()
        if code_lines:
            self.render_code_block("\n".join(code_lines))

    # ---------- 行内格式 ----------
    def _parse_inline(self, text: str):
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

        segments = []
        idx = 0
        for m in re.finditer(r"(\*\*(.+?)\*\*|`([^`]+)`)", text):
            start, end = m.span()
            if start > idx:
                segments.append(("normal", text[idx:start]))
            if m.group(1).startswith("**"):
                segments.append(("bold", m.group(2)))
            else:
                segments.append(("code", m.group(3)))
            idx = end
        if idx < len(text):
            segments.append(("normal", text[idx:]))
        if not segments:
            segments.append(("normal", text))
        return segments

    def _set_font_by_segment(self, kind: str, base_size: int):
        if kind == "bold":
            self.set_font("CN", "B", base_size)
        elif kind == "code":
            self.set_font("Courier", "", max(8, base_size - 2))
        else:
            self.set_font("CN", "", base_size)

    # ---------- 元素渲染 ----------
    def render_header(self, level: int, text: str):
        sizes = {1: 17, 2: 14, 3: 12, 4: 11, 5: 10, 6: 10}
        size = sizes.get(level, 11)
        self.ln(self.base_line_height * 1.0)
        self.set_text_color(0, 0, 0)
        self.set_x(self.content_x)
        segments = self._parse_inline(text)
        for kind, txt in segments:
            self._set_font_by_segment(kind, size)
            self.write(self.base_line_height * (size / 11), txt)
        self.ln(self.base_line_height * 1.0)
        if level <= 2:
            self.set_draw_color(0, 0, 0)
            y = self.get_y()
            self.line(self.content_x, y, self.content_x + self.content_width, y)
            self.ln(self.base_line_height * 0.3)

    def render_paragraph(self, text: str):
        self.set_text_color(30, 30, 30)
        self.set_x(self.content_x)
        segments = self._parse_inline(text)
        for kind, txt in segments:
            self._set_font_by_segment(kind, 11)
            self.multi_cell(self.content_width, self.base_line_height, txt, align="J")
            self.set_x(self.content_x)
        self.ln(self.base_line_height * 0.3)

    def render_list_item(self, text: str, bullet: str):
        self.set_text_color(30, 30, 30)
        bullet_width = 8
        self.set_x(self.content_x)
        self.set_font("CN", "B", 11)
        self.cell(bullet_width, self.base_line_height, bullet, new_x="RIGHT", new_y="TOP")

        segments = self._parse_inline(text)
        x_after_bullet = self.content_x + bullet_width
        self.set_x(x_after_bullet)
        for kind, txt in segments:
            self._set_font_by_segment(kind, 11)
            self.multi_cell(self.content_width - bullet_width, self.base_line_height, txt, align="J")
            self.set_x(x_after_bullet)
        self.ln(self.base_line_height * 0.15)

    def render_hr(self):
        self.ln(self.base_line_height * 0.4)
        self.set_draw_color(180, 180, 180)
        y = self.get_y()
        margin = 15
        self.line(self.content_x + margin, y, self.content_x + self.content_width - margin, y)
        self.ln(self.base_line_height * 0.4)

    def render_blockquote(self, text: str):
        pad = 4
        box_x = self.content_x
        box_w = self.content_width
        self.set_font("CN", "", 10)
        self.set_text_color(60, 60, 60)
        # 估算高度
        h = self.base_line_height * (text.count("\n") + 1) + pad * 2
        self.set_fill_color(240, 248, 255)
        self.set_draw_color(70, 130, 180)
        self.rect(box_x, self.get_y(), box_w, h, style="FD")
        self.set_xy(box_x + pad, self.get_y() + pad)
        self.multi_cell(box_w - pad * 2, self.base_line_height * 1.05, text, align="L")
        self.ln(self.base_line_height * 0.3)
        self.set_text_color(30, 30, 30)

    def render_code_block(self, text: str):
        pad = 4
        box_x = self.content_x
        box_w = self.content_width
        self.set_font("CN", "", 8)
        self.set_text_color(50, 50, 50)
        lines = text.splitlines() or [""]
        line_h = 4.5
        h = len(lines) * line_h + pad * 2
        self.set_fill_color(245, 245, 245)
        self.rect(box_x, self.get_y(), box_w, h, style="F")
        self.set_xy(box_x + pad, self.get_y() + pad)
        for line in lines:
            self.cell(0, line_h, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(self.base_line_height * 0.3)

    def render_table(self, rows: list):
        if len(rows) < 3:
            return
        cells = []
        for r in rows:
            row_cells = [c.strip() for c in r.split("|")]
            # 去掉首尾空单元格
            if row_cells and row_cells[0] == "":
                row_cells = row_cells[1:]
            if row_cells and row_cells[-1] == "":
                row_cells = row_cells[:-1]
            cells.append(row_cells)

        header = cells[0]
        data = cells[2:]
        col_count = len(header)
        if col_count == 0:
            return
        col_width = self.content_width / col_count

        # 字体大小根据列数调整
        font_size = 9 if col_count <= 4 else 7
        self.set_text_color(0, 0, 0)

        def cell_height(row):
            self.set_font("CN", "", font_size)
            h = self.base_line_height
            for c in row:
                lines = max(1, (len(c) * 2.5) // col_width + 1)
                h = max(h, self.base_line_height * lines)
            return h

        self.ln(self.base_line_height * 0.3)
        start_x = self.content_x

        # 表头
        self.set_fill_color(230, 230, 230)
        self.set_font("CN", "B", font_size)
        h = cell_height(header)
        self.set_x(start_x)
        for c in header:
            self.cell(col_width, h, c, border=1, align="C", fill=True)
        self.ln(h)

        # 数据行
        self.set_fill_color(255, 255, 255)
        for row in data:
            h = cell_height(row)
            self.set_x(start_x)
            for c in row:
                self.set_font("CN", "", font_size)
                # multi_cell 绘制带自动换行的单元格
                x = self.get_x()
                y = self.get_y()
                self.multi_cell(col_width, self.base_line_height, c, border=1, align="C")
                # 回到同行下一个单元格位置
                self.set_xy(x + col_width, y)
            self.ln(h)
        self.ln(self.base_line_height * 0.3)


def main():
    if len(sys.argv) < 3:
        print("Usage: python md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    md_text = input_path.read_text(encoding="utf-8")

    font_regular = r"C:\Windows\Fonts\msyh.ttc"
    font_bold = r"C:\Windows\Fonts\msyhbd.ttc"

    pdf = MarkdownPDF(font_regular, font_bold)

    # 封面
    pdf.add_page()
    pdf.set_font("CN", "B", 26)
    pdf.set_y(80)
    pdf.cell(0, 15, "mt-embodied-sim", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CN", "B", 20)
    pdf.cell(0, 15, "商业计划书", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("CN", "", 12)
    pdf.cell(0, 10, "面向家庭5S隐形家务的国产具身智能仿真平台", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("CN", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "版本：v1.1（已融合2026专项行动政策解读）", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "日期：2026年6月12日", align="C", new_x="LMARGIN", new_y="NEXT")

    # 正文
    pdf.add_page()
    pdf.render_markdown(md_text)

    pdf.output(str(output_path))
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    main()

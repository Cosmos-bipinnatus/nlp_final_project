# -*- coding: utf-8 -*-
"""
pdf_generator.py
已重構：
1. 採用 CJK (Droid Sans Fallback) 嵌入式字型，在每一頁創建時動態調用 insert_font 將字型 buffer 物理嵌入 PDF 中，
   根本解決在手機、Mac 等平台的「need font file or buffer」與中文空白/亂碼問題。
2. 採用「視覺字元寬度折行演算法」（ASCII寬度=1，非ASCII/中文寬度=2），解決英中混排早折行或溢出邊界的問題。
"""
import fitz
import re
from datetime import datetime
from typing import List, Dict, Any

def clean_html(text: str) -> str:
    """
    清除 HTML 標籤，保留換行與基本格式。
    """
    if not text:
        return ""
    # 移除 CSS / Style 區塊
    text = re.sub(r'<style>.*?</style>', '', text, flags=re.DOTALL)
    # 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    # 替換多個換行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def wrap_text(text: str, max_visual_width: int) -> List[str]:
    """
    根據字元視覺寬度（ASCII為1，非ASCII/中文為2）對字串進行智慧折行。
    """
    wrapped_lines = []
    for line in text.split("\n"):
        if not line:
            wrapped_lines.append("")
            continue
            
        current_line = []
        current_width = 0
        
        for char in line:
            # 判斷是否為 ASCII 字元 (半寬計為 1，全寬計為 2)
            char_width = 1 if ord(char) < 128 else 2
            
            if current_width + char_width > max_visual_width:
                wrapped_lines.append("".join(current_line))
                current_line = [char]
                current_width = char_width
            else:
                current_line.append(char)
                current_width += char_width
                
        if current_line:
            wrapped_lines.append("".join(current_line))
            
    return wrapped_lines

def add_page(doc: fitz.Document, width: float, height: float, fontname: str, font_buffer: bytes) -> fitz.Page:
    """
    創建新頁面並立即將 CJK 字型緩衝區嵌入該頁面資源中。
    這是解決 "need font file or buffer" 的關鍵！
    """
    page = doc.new_page(width=width, height=height)
    page.insert_font(fontname=fontname, fontbuffer=font_buffer)
    return page

def generate_qa_pdf(chat_history: List[Dict[str, Any]]) -> bytes:
    """
    將對話歷史轉換為排版精美的 PDF 二進位資料。
    """
    doc = fitz.open()
    cjk_font = fitz.Font("cjk")
    font_buffer = cjk_font.buffer
    fontname = "cjk"
    
    # 頁面規格 (A4)
    page_w = 595.3
    page_h = 841.9
    margin = 50
    usable_w = page_w - 2 * margin
    
    # 字體大小與行高
    title_size = 18
    h2_size = 13
    body_size = 10
    line_h = 15
    chars_per_line = 90  # 基於視覺寬度的每行上限 (等同 90 個半寬英文或 45 個全寬中文)
    
    page = add_page(doc, page_w, page_h, fontname, font_buffer)
    y = margin
    
    # 寫入主標題
    page.insert_text((margin, y), "📚 Literature Reviewer 學術問答報告", fontsize=title_size, fontname=fontname, color=(0.02, 0.45, 0.55))
    y += 25
    page.insert_text((margin, y), f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=8, fontname=fontname, color=(0.5, 0.5, 0.5))
    y += 25
    
    # 畫一條分隔線
    page.draw_line(fitz.Point(margin, y), fitz.Point(page_w - margin, y), color=(0.02, 0.45, 0.55), width=1)
    y += 25
    
    for idx, msg in enumerate(chat_history):
        role = msg.get("role", "user")
        
        if role == "user":
            heading = f"提問 #{idx // 2 + 1}："
            content = msg.get("content", "")
            color_heading = (0.06, 0.45, 0.55)  # 深藍
            color_text = (0.1, 0.1, 0.1)
        else:
            heading = "AI 學術導師分析："
            content = msg.get("plain_answer") or msg.get("content", "")
            content = clean_html(content)
            color_heading = (0.08, 0.57, 0.67)  # 淺藍
            color_text = (0.2, 0.2, 0.2)
            
        # 寫入角色標題
        # 檢查是否需要分頁
        if y + 25 > page_h - margin:
            page = add_page(doc, page_w, page_h, fontname, font_buffer)
            y = margin + 20
            
        page.insert_text((margin, y), heading, fontsize=h2_size, fontname=fontname, color=color_heading)
        y += 18
        
        # 寫入內容文字
        lines = wrap_text(content, chars_per_line)
        for line in lines:
            if y > page_h - margin - 20:
                page = add_page(doc, page_w, page_h, fontname, font_buffer)
                y = margin + 20
            page.insert_text((margin, y), line, fontsize=body_size, fontname=fontname, color=color_text)
            y += line_h
            
        # 寫入參考來源（若是 Assistant）
        if role == "assistant" and msg.get("sources"):
            y += 5
            if y + 15 > page_h - margin:
                page = add_page(doc, page_w, page_h, fontname, font_buffer)
                y = margin + 20
            page.insert_text((margin, y), "📄 參考文獻來源：", fontsize=body_size - 1, fontname=fontname, color=(0.3, 0.6, 0.5))
            y += line_h
            
            for s_html in msg["sources"]:
                s_clean = clean_html(s_html)
                s_lines = wrap_text(f"- {s_clean}", chars_per_line)
                for s_line in s_lines:
                    if y > page_h - margin - 20:
                        page = add_page(doc, page_w, page_h, fontname, font_buffer)
                        y = margin + 20
                    page.insert_text((margin + 10, y), s_line, fontsize=body_size - 1.5, fontname=fontname, color=(0.4, 0.4, 0.4))
                    y += line_h - 2
                    
        y += 20  # 訊息間隔
        
    # 最後統一在每頁加上頁碼與裝飾線
    total_pages = len(doc)
    for p_idx in range(total_pages):
        p = doc[p_idx]
        # 頁首線
        if p_idx > 0:
            p.draw_line(fitz.Point(margin, 35), fitz.Point(page_w - margin, 35), color=(0.8, 0.8, 0.8), width=0.5)
        # 頁尾線
        p.draw_line(fitz.Point(margin, page_h - 35), fitz.Point(page_w - margin, page_h - 35), color=(0.8, 0.8, 0.8), width=0.5)
        # 頁尾頁碼
        p.insert_text((page_w / 2 - 10, page_h - 20), f"- {p_idx + 1} / {total_pages} -", fontsize=8, fontname=fontname, color=(0.5, 0.5, 0.5))
        
    return doc.tobytes()

def generate_comparison_pdf(comparison_data: List[Dict[str, Any]]) -> bytes:
    """
    將跨文獻比較資料轉換為結構化排版 PDF 二進位資料。
    """
    doc = fitz.open()
    cjk_font = fitz.Font("cjk")
    font_buffer = cjk_font.buffer
    fontname = "cjk"
    
    page_w = 595.3
    page_h = 841.9
    margin = 50
    usable_w = page_w - 2 * margin
    
    title_size = 18
    h2_size = 13
    label_size = 10
    body_size = 10
    line_h = 15
    chars_per_line = 85  # 縮排後基於視覺寬度的每行上限
    
    page = add_page(doc, page_w, page_h, fontname, font_buffer)
    y = margin
    
    # 寫入主標題
    page.insert_text((margin, y), "📊 Literature Reviewer 跨文獻學術比較報告", fontsize=title_size, fontname=fontname, color=(0.02, 0.45, 0.55))
    y += 25
    page.insert_text((margin, y), f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=8, fontname=fontname, color=(0.5, 0.5, 0.5))
    y += 25
    
    # 畫一條分隔線
    page.draw_line(fitz.Point(margin, y), fitz.Point(page_w - margin, y), color=(0.02, 0.45, 0.55), width=1)
    y += 25
    
    for idx, paper in enumerate(comparison_data, 1):
        # 寫入論文大標題
        if y + 40 > page_h - margin:
            page = add_page(doc, page_w, page_h, fontname, font_buffer)
            y = margin + 20
            
        paper_title = paper.get("title") or paper.get("pdf_file") or "未知論文"
        page.insert_text((margin, y), f"📖 文獻 #{idx}：{paper_title}", fontsize=h2_size, fontname=fontname, color=(0.02, 0.45, 0.55))
        y += 20
        
        # 畫底線
        page.draw_line(fitz.Point(margin, y - 2), fitz.Point(margin + 150, y - 2), color=(0.02, 0.45, 0.55), width=0.5)
        
        # 定義要提取的欄位
        sections = [
            ("📄 原始 PDF 檔名", paper.get("pdf_file", "")),
            ("🛠️ 核心研究方法", paper.get("methodology", "")),
            ("📊 實驗資料集與指標", paper.get("datasets", "")),
            ("👍 主要創新與優點", paper.get("pros", "")),
            ("👎 局限性與缺點", paper.get("cons", ""))
        ]
        
        for label, val in sections:
            if y + 25 > page_h - margin:
                page = add_page(doc, page_w, page_h, fontname, font_buffer)
                y = margin + 20
                
            # 寫入欄位標籤
            page.insert_text((margin, y), label, fontsize=label_size, fontname=fontname, color=(0.1, 0.5, 0.6))
            y += 14
            
            # 寫入欄位內容
            val_clean = clean_html(val)
            lines = wrap_text(val_clean, chars_per_line)
            for line in lines:
                if y > page_h - margin - 20:
                    page = add_page(doc, page_w, page_h, fontname, font_buffer)
                    y = margin + 20
                page.insert_text((margin + 15, y), line, fontsize=body_size, fontname=fontname, color=(0.2, 0.2, 0.2))
                y += line_h
            y += 8  # 欄位間隔
            
        y += 15  # 論文間隔
        
    # 最後統一在每頁加上頁碼與裝飾線
    total_pages = len(doc)
    for p_idx in range(total_pages):
        p = doc[p_idx]
        # 頁首線
        if p_idx > 0:
            p.draw_line(fitz.Point(margin, 35), fitz.Point(page_w - margin, 35), color=(0.8, 0.8, 0.8), width=0.5)
        # 頁尾線
        p.draw_line(fitz.Point(margin, page_h - 35), fitz.Point(page_w - margin, page_h - 35), color=(0.8, 0.8, 0.8), width=0.5)
        # 頁尾頁碼
        p.insert_text((page_w / 2 - 10, page_h - 20), f"- {p_idx + 1} / {total_pages} -", fontsize=8, fontname=fontname, color=(0.5, 0.5, 0.5))
        
    return doc.tobytes()

def add_review_page(doc: fitz.Document, width: float, height: float, fontname: str, font_buffer: bytes, margin: float) -> fitz.Page:
    """
    為綜述報告量身設計的頁面創建與裝飾條繪製方法。
    """
    page = doc.new_page(width=width, height=height)
    page.insert_font(fontname=fontname, fontbuffer=font_buffer)
    # 畫一條左側學術藍裝飾條
    page.draw_rect(fitz.Rect(margin - 15, margin, margin - 10, height - margin), color=(0.02, 0.45, 0.55), fill=(0.02, 0.45, 0.55))
    return page

def is_separator_row(row_str: str) -> bool:
    """
    檢查該行是否為 Markdown 表格的分隔線列（例如 | :--- | :--- | 或 | --- | --- |）
    """
    row_str = row_str.strip()
    if not row_str.startswith("|") or not row_str.endswith("|"):
        return False
    # 分隔線列應該只包含 |、-、:、空格
    chars = set(row_str.replace(" ", "").replace("-", "").replace(":", "").replace("|", ""))
    return len(chars) == 0

def render_markdown_table(doc: fitz.Document, page: fitz.Page, table_lines: List[str], 
                          fontname: str, font_buffer: bytes, margin: float, 
                          page_w: float, page_h: float, y: float) -> tuple[fitz.Page, float]:
    """
    將 Markdown 表格解析並在 PDF 頁面中繪製出漂亮的框線表格，支援折行與跨頁分頁。
    """
    rows = []
    for line in table_lines:
        # 以 | 分割，移除首尾空字串
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            rows.append(parts[1:-1])
            
    if len(rows) < 2:
        return page, y
        
    headers = rows[0]
    # 排除 rows[1] (分隔線列)
    data_rows = rows[2:] if len(rows) > 2 else []
    
    num_cols = len(headers)
    if num_cols == 0:
        return page, y
        
    usable_w = page_w - 2 * margin
    
    # 決定欄位寬度分配
    # 第一欄（通常為「特徵」或「指標」）分配較小比例（22%），其餘平分
    if num_cols == 1:
        col_widths = [usable_w]
        other_col_w = usable_w
    else:
        first_col_w = usable_w * 0.22
        other_col_w = (usable_w - first_col_w) / (num_cols - 1)
        col_widths = [first_col_w] + [other_col_w] * (num_cols - 1)
        
    table_font_size = 8.5
    cell_padding_x = 6
    cell_padding_y = 6
    table_line_h = 12
    
    def get_row_cells_and_height(row_cells):
        wrapped_cells = []
        max_lines = 1
        for col_idx, cell_text in enumerate(row_cells):
            col_w = col_widths[col_idx] if col_idx < len(col_widths) else other_col_w
            # 計算該欄寬度能容納的視覺字元長度
            max_vis_w = int((col_w - 2 * cell_padding_x) / (table_font_size * 0.5))
            if max_vis_w < 5:
                max_vis_w = 5
            # 使用 pdf_generator 中已有的 wrap_text 進行中文/英文折行
            wrapped = wrap_text(cell_text, max_vis_w)
            wrapped_cells.append(wrapped)
            if len(wrapped) > max_lines:
                max_lines = len(wrapped)
        row_height = max_lines * table_line_h + 2 * cell_padding_y
        return wrapped_cells, row_height
        
    # 計算表頭高度
    header_wrapped, header_h = get_row_cells_and_height(headers)
    
    def draw_row(tgt_page, y_pos, wrapped_cells, row_h, is_header=False):
        # 繪製背景色（表頭用淡藍色背景）
        if is_header:
            bg_color = (0.90, 0.94, 0.96)
            tgt_page.draw_rect(fitz.Rect(margin, y_pos, page_w - margin, y_pos + row_h), color=bg_color, fill=bg_color)
            
        # 逐格繪製文字
        x_pos = margin
        for col_idx, cell_lines in enumerate(wrapped_cells):
            col_w = col_widths[col_idx] if col_idx < len(col_widths) else other_col_w
            
            line_y = y_pos + cell_padding_y + table_font_size - 1
            for line_text in cell_lines:
                tgt_page.insert_text(
                    (x_pos + cell_padding_x, line_y),
                    line_text,
                    fontsize=table_font_size,
                    fontname=fontname,
                    color=(0.02, 0.35, 0.45) if is_header else (0.2, 0.2, 0.2)
                )
                line_y += table_line_h
            x_pos += col_w
            
        # 繪製單列的水平格線
        tgt_page.draw_line(fitz.Point(margin, y_pos), fitz.Point(page_w - margin, y_pos), color=(0.8, 0.8, 0.8), width=0.5)
        tgt_page.draw_line(fitz.Point(margin, y_pos + row_h), fitz.Point(page_w - margin, y_pos + row_h), color=(0.8, 0.8, 0.8), width=0.5)
        
        # 繪製垂直格線
        x_pos = margin
        tgt_page.draw_line(fitz.Point(x_pos, y_pos), fitz.Point(x_pos, y_pos + row_h), color=(0.8, 0.8, 0.8), width=0.5)
        for col_w in col_widths:
            x_pos += col_w
            tgt_page.draw_line(fitz.Point(x_pos, y_pos), fitz.Point(x_pos, y_pos + row_h), color=(0.8, 0.8, 0.8), width=0.5)
            
    # 檢查表頭是否需要換頁
    if y + header_h > page_h - margin - 20:
        page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
        y = margin
        
    # 繪製表頭
    draw_row(page, y, header_wrapped, header_h, is_header=True)
    y += header_h
    
    # 繪製資料列
    for row_cells in data_rows:
        # 長度不足則補齊
        if len(row_cells) < num_cols:
            row_cells += [""] * (num_cols - len(row_cells))
        elif len(row_cells) > num_cols:
            row_cells = row_cells[:num_cols]
            
        wrapped_cells, row_h = get_row_cells_and_height(row_cells)
        
        # 換頁檢測
        if y + row_h > page_h - margin - 20:
            page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
            y = margin
            # 換頁後重新繪製表頭，確保閱讀流暢
            draw_row(page, y, header_wrapped, header_h, is_header=True)
            y += header_h
            
        draw_row(page, y, wrapped_cells, row_h, is_header=False)
        y += row_h
        
    return page, y + 10

def generate_review_pdf(report_text: str) -> bytes:
    """
    將 Markdown 格式的文獻回顧綜述報告轉換為排版精美的 PDF 二進位資料。
    """
    doc = fitz.open()
    cjk_font = fitz.Font("cjk")
    font_buffer = cjk_font.buffer
    fontname = "cjk"
    
    page_w = 595.3
    page_h = 841.9
    margin = 50
    usable_w = page_w - 2 * margin
    
    title_size = 18
    h2_size = 13
    body_size = 10
    line_h = 16
    chars_per_line = 90  # 基於視覺寬度的每行上限
    
    page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
    y = margin
    
    lines = report_text.split("\n")
    idx = 0
    num_lines = len(lines)
    
    while idx < num_lines:
        line = lines[idx]
        line_str = line.strip()
        
        # 偵測是否為 Markdown 表格
        if line_str.startswith("|") and idx + 1 < num_lines and is_separator_row(lines[idx+1]):
            table_lines = []
            while idx < num_lines and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            # 繪製表格並更新 page 與 y
            page, y = render_markdown_table(doc, page, table_lines, fontname, font_buffer, margin, page_w, page_h, y)
            continue
            
        if not line_str:
            y += 8
            idx += 1
            continue
            
        # 處理 Markdown 標題與格式
        if line_str.startswith("# "):
            text = line_str[2:].strip().replace("**", "")
            if y + 35 > page_h - margin:
                page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
                y = margin
            page.insert_text((margin, y), text, fontsize=title_size, fontname=fontname, color=(0.02, 0.45, 0.55))
            y += 30
            # 畫底線
            page.draw_line(fitz.Point(margin, y - 5), fitz.Point(page_w - margin, y - 5), color=(0.02, 0.45, 0.55), width=1)
            y += 10
            
        elif line_str.startswith("## "):
            text = line_str[3:].strip().replace("**", "")
            if y + 28 > page_h - margin:
                page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
                y = margin
            y += 10  # 標題前留白
            page.insert_text((margin, y), text, fontsize=h2_size, fontname=fontname, color=(0.02, 0.45, 0.55))
            y += 20
            
        elif line_str.startswith("### "):
            text = line_str[4:].strip().replace("**", "")
            if y + 24 > page_h - margin:
                page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
                y = margin
            page.insert_text((margin, y), text, fontsize=body_size + 1, fontname=fontname, color=(0.06, 0.55, 0.65))
            y += 18
            
        elif line_str.startswith("- ") or line_str.startswith("* "):
            text = line_str[2:].strip().replace("**", "")
            wrapped_lines = wrap_text(f"•  {text}", chars_per_line)
            for w_line in wrapped_lines:
                if y > page_h - margin - 20:
                    page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
                    y = margin
                page.insert_text((margin + 10, y), w_line, fontsize=body_size, fontname=fontname, color=(0.2, 0.2, 0.2))
                y += line_h
                
        elif re.match(r'^\d+\.\s', line_str):
            match = re.match(r'^(\d+\.\s)(.*)', line_str)
            prefix = match.group(1)
            text = match.group(2).strip().replace("**", "")
            wrapped_lines = wrap_text(f"{prefix}{text}", chars_per_line)
            for w_line in wrapped_lines:
                if y > page_h - margin - 20:
                    page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
                    y = margin
                page.insert_text((margin, y), w_line, fontsize=body_size, fontname=fontname, color=(0.2, 0.2, 0.2))
                y += line_h
                
        else:
            text = line_str.replace("**", "")
            indent = 0
            if text.startswith(">"):
                text = text[1:].strip()
                indent = 15
                
            wrapped_lines = wrap_text(text, chars_per_line - (indent // 10))
            for w_line in wrapped_lines:
                if y > page_h - margin - 20:
                    page = add_review_page(doc, page_w, page_h, fontname, font_buffer, margin)
                    y = margin
                page.insert_text((margin + indent, y), w_line, fontsize=body_size, fontname=fontname, color=(0.2, 0.2, 0.2))
                y += line_h
                
        idx += 1
        
    # 最後統一在每頁加上頁碼與裝飾線
    total_pages = len(doc)
    for p_idx in range(total_pages):
        p = doc[p_idx]
        # 頁首線
        if p_idx > 0:
            p.draw_line(fitz.Point(margin, 35), fitz.Point(page_w - margin, 35), color=(0.8, 0.8, 0.8), width=0.5)
        # 頁尾線
        p.draw_line(fitz.Point(margin, page_h - 35), fitz.Point(page_w - margin, page_h - 35), color=(0.8, 0.8, 0.8), width=0.5)
        # 頁尾頁碼
        p.insert_text((page_w / 2 - 10, page_h - 20), f"- {p_idx + 1} / {total_pages} -", fontsize=8, fontname=fontname, color=(0.5, 0.5, 0.5))
        
    return doc.tobytes()

# -*- coding: utf-8 -*-
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

def wrap_text(text: str, chars_per_line: int) -> List[str]:
    """
    按固定字數對字串進行折行（因 china-t 為等寬 CJK 字型）。
    """
    wrapped_lines = []
    for line in text.split("\n"):
        if not line:
            wrapped_lines.append("")
            continue
        
        # 進行按字數切割
        i = 0
        while i < len(line):
            wrapped_lines.append(line[i : i + chars_per_line])
            i += chars_per_line
    return wrapped_lines

def generate_qa_pdf(chat_history: List[Dict[str, Any]]) -> bytes:
    """
    將對話歷史轉換為排版精美的 PDF 二進位資料。
    """
    doc = fitz.open()
    fontname = "china-t"
    
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
    chars_per_line = int(usable_w / body_size) # 每行字數 (約 49 字)
    
    page = doc.new_page(width=page_w, height=page_h)
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
            color_heading = (0.06, 0.45, 0.55) # 深藍
            color_text = (0.1, 0.1, 0.1)
        else:
            heading = "AI 學術導師分析："
            content = msg.get("plain_answer") or msg.get("content", "")
            # 清除 HTML
            content = clean_html(content)
            color_heading = (0.08, 0.57, 0.67) # 淺藍
            color_text = (0.2, 0.2, 0.2)
            
        # 寫入角色標題
        # 檢查是否需要分頁
        if y + 25 > page_h - margin:
            page = doc.new_page(width=page_w, height=page_h)
            y = margin + 20
            
        page.insert_text((margin, y), heading, fontsize=h2_size, fontname=fontname, color=color_heading)
        y += 18
        
        # 寫入內容文字
        lines = wrap_text(content, chars_per_line)
        for line in lines:
            if y > page_h - margin - 20:
                page = doc.new_page(width=page_w, height=page_h)
                y = margin + 20
            page.insert_text((margin, y), line, fontsize=body_size, fontname=fontname, color=color_text)
            y += line_h
            
        # 寫入參考來源（若是 Assistant）
        if role == "assistant" and msg.get("sources"):
            y += 5
            if y + 15 > page_h - margin:
                page = doc.new_page(width=page_w, height=page_h)
                y = margin + 20
            page.insert_text((margin, y), "📄 參考文獻來源：", fontsize=body_size - 1, fontname=fontname, color=(0.3, 0.6, 0.5))
            y += line_h
            
            for s_html in msg["sources"]:
                s_clean = clean_html(s_html)
                s_lines = wrap_text(f"- {s_clean}", chars_per_line)
                for s_line in s_lines:
                    if y > page_h - margin - 20:
                        page = doc.new_page(width=page_w, height=page_h)
                        y = margin + 20
                    page.insert_text((margin + 10, y), s_line, fontsize=body_size - 1.5, fontname=fontname, color=(0.4, 0.4, 0.4))
                    y += line_h - 2
                    
        y += 20 # 訊息間隔
        
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
    fontname = "china-t"
    
    page_w = 595.3
    page_h = 841.9
    margin = 50
    usable_w = page_w - 2 * margin
    
    title_size = 18
    h2_size = 13
    label_size = 10
    body_size = 10
    line_h = 15
    chars_per_line = int(usable_w / body_size)
    
    page = doc.new_page(width=page_w, height=page_h)
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
            page = doc.new_page(width=page_w, height=page_h)
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
                page = doc.new_page(width=page_w, height=page_h)
                y = margin + 20
                
            # 寫入欄位標籤
            page.insert_text((margin, y), label, fontsize=label_size, fontname=fontname, color=(0.1, 0.5, 0.6))
            y += 14
            
            # 寫入欄位內容
            val_clean = clean_html(val)
            lines = wrap_text(val_clean, chars_per_line)
            for line in lines:
                if y > page_h - margin - 20:
                    page = doc.new_page(width=page_w, height=page_h)
                    y = margin + 20
                page.insert_text((margin + 15, y), line, fontsize=body_size, fontname=fontname, color=(0.2, 0.2, 0.2))
                y += line_h
            y += 8 # 欄位間隔
            
        y += 15 # 論文間隔
        
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

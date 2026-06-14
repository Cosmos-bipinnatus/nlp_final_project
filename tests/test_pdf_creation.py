# -*- coding: utf-8 -*-
"""
test_pdf_creation.py
驗證 PDF 生成器是否能成功將測試資料輸出為 PDF，確保不再拋出 'need font file or buffer' 異常。
"""
import sys
from pathlib import Path

# 將專案根目錄加入路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 避免 Windows 主機 console 輸出不支援的 UTF-8 字元而崩潰
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.utils.pdf_generator import generate_review_pdf

def test_pdf_generation():
    print("=== 開始測試 PDF 生成器 ===")
    test_report = """# 📚 測試學術報告
## 1. 🎯 研究背景
這是一篇測試中文 PDF 生成的範本報告。

## 2. 🔬 核心方法
本系統採用 Google Gemini Embeddings 搭配 ChromaDB 進行語意檢索，測試中文與英文折行狀況。
We test how English words like Attention and Transformer wrap alongside Chinese characters.
"""
    try:
        pdf_bytes = generate_review_pdf(test_report)
        print(f"✅ PDF 生成成功！位元組大小: {len(pdf_bytes)} bytes")
        
        # 寫入測試檔案
        output_path = Path(__file__).resolve().parent / "test_output.pdf"
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"✅ 成功將測試 PDF 寫入至: {output_path}")
        
    except Exception as e:
        print(f"❌ PDF 生成失敗！異常原因: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pdf_generation()

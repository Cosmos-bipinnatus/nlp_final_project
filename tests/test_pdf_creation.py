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

| 特徵 | 文獻 #1 (MMBERT) | 文獻 #2 (BERT/RoBERTa) |
| :--- | :--- | :--- |
| 主要模態 | 多模態 (文本、語音、視覺) | 單模態 (文本) |
| 核心架構創新 | MoE架構、三階段漸進式訓練、輕量級路由模組 | 標準Transformer模型微調 |
| 魯棒性策略 | MoE動態專家激活、跨模態整合以對抗偽裝擾動 | 透過平衡資料集、強健的預訓練模型提升韌性 |
| 資料處理重點 | 多模態對齊、跨模態表示學習 | 資料集平衡、精細化文本預處理、社會語言學特徵考量 |
| 模型比較 | 內部MoE專家動態調整 | BERT與RoBERTa的直接性能比較 |
| 計算效率考量 | MoE架構提升計算效率與靈活性 (稀疏激活) | 透過選擇更優模型 (RoBERTa) 提升效能 |
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

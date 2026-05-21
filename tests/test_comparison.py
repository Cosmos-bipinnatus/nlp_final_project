# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# 將 src/ 目錄加入 Python 模組搜尋路徑，確保可以順利 import src 中的模組
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.comparison_manager import AcademicComparisonManager

def test_comparison_extraction():
    # 處理 Windows CLI 編碼問題，忽略無法在 cp950 下顯示的 Emoji 等字元
    sys.stdout.reconfigure(errors='ignore') if hasattr(sys.stdout, 'reconfigure') else None
    
    print("[Info] Starting Literature Comparison Matrix Unit Test...")
    
    # 1. 尋找 data/ 資料夾下的 PDF 論文
    data_dir = Path("data")
    existing_pdfs = list(data_dir.glob("*.pdf"))
    
    if not existing_pdfs:
        print("[Error] No PDF files found in data/ directory. Please upload at least one PDF.")
        return
        
    target_pdf = existing_pdfs[0].name
    print(f"[Info] Found test document in data/: '{target_pdf}'")
    
    # 2. 初始化比較管理員
    try:
        print("\n--- Step 1: Initializing AcademicComparisonManager ---")
        manager = AcademicComparisonManager(persist_directory="vectorstore")
        print("[Success] Comparison Manager initialized successfully.")
        
        # 3. 測試特定論文的特徵提取
        print(f"\n--- Step 2: Extracting Features for '{target_pdf}' ---")
        print("[Processing] Fetching topic-specific context and calling Gemini Structured LLM...")
        features = manager.extract_features_for_paper(target_pdf)
        
        # 4. 展示與驗證結構化輸出
        print("\n" + "="*80)
        print("[Results] Pydantic Structured Paper Features Extracted:")
        print("="*80)
        print(f"📌 [Title]       : {features.title}")
        print(f"🛠️ [Methodology] : {features.methodology}")
        print(f"📊 [Datasets]    : {features.datasets}")
        print(f"👍 [Pros/贡献]   : {features.pros}")
        print(f"👎 [Cons/局限]   : {features.cons}")
        print("="*80)
        
        # 5. 測試轉換為 Markdown 表格
        print("\n--- Step 3: Verifying Markdown Table Conversion ---")
        mock_list = [{
            "pdf_file": target_pdf,
            "title": features.title,
            "methodology": features.methodology,
            "datasets": features.datasets,
            "pros": features.pros,
            "cons": features.cons
        }]
        md_table = manager.convert_to_markdown_table(mock_list)
        print("[Success] Markdown Table generated successfully:")
        print(md_table)
        
        print("\n✅ Week 6 Literature Comparison Grid Ingestion Test Passed successfully!")
        
    except Exception as e:
        print(f"[Error] Comparison Matrix unit test failed: {e}")

if __name__ == "__main__":
    test_comparison_extraction()

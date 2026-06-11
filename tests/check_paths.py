# -*- coding: utf-8 -*-
"""
專案路徑與環境變數健康檢查工具 (check_paths.py)
用於引導大二學生在本地運行測試前，快速健檢路徑配置與 .env 設定。
"""

import os
import sys
from pathlib import Path

# 將專案根目錄加入 Python 模組搜尋路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 避免 Windows 主機 console 輸出不支援的 UTF-8 字元而崩潰
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_diagnostics() -> None:
    print("==================================================")
    print(" 🔍 Literature Reviewer 專案路徑與環境健檢工具")
    print("==================================================")

    # 1. 檢查 GEMINI_API_KEY 環境變數
    print("\n[Step 1] 檢查 GEMINI_API_KEY 環境變數")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ 未偵測到 GEMINI_API_KEY 環境變數！")
        print("   👉 提示: 在本地運行 CLI 測試前，請先在終端機中設定臨時環境變數。")
        print("      PowerShell (Windows): $env:GEMINI_API_KEY=\"您的金鑰\"")
        print("      Bash (Linux/Mac): export GEMINI_API_KEY=\"您的金鑰\"")
    else:
        print(f"✅ GEMINI_API_KEY 環境變數已載入 (長度: {len(api_key)} 字元)")

    # 2. 載入並檢查 settings.py 的路徑定義
    print("\n[Step 2] 驗證專案路徑配置 (pathlib)")
    try:
        from src.config.settings import (
            BASE_DIR,
            DATA_DIR,
            DOCS_DIR,
            VECTORSTORE_DIR,
            MODEL_NAME,
            EMBEDDING_MODEL
        )
        print(f"✅ 成功載入 src.config.settings")
        print(f"   - 專案根目錄 (BASE_DIR)      : {BASE_DIR}")
        print(f"   - 論文 PDF 目錄 (DATA_DIR)    : {DATA_DIR} (存在: {DATA_DIR.exists()})")
        print(f"   - 報告輸出目錄 (DOCS_DIR)    : {DOCS_DIR} (存在: {DOCS_DIR.exists()})")
        print(f"   - 向量庫目錄 (VECTORSTORE_DIR): {VECTORSTORE_DIR} (存在: {VECTORSTORE_DIR.exists()})")
    except Exception as e:
        print(f"❌ 載入路徑配置時發生錯誤: {e}")
        return

    # 3. 檢查測試 PDF 檔案
    print("\n[Step 3] 檢查本地測試論文數據")
    target_test_pdf = DATA_DIR / "Attentation Is All You Need.pdf"
    
    # 搜尋 data 目錄下所有 pdf
    existing_pdfs = list(DATA_DIR.glob("*.pdf"))
    print(f"   - 目前 data/ 目錄下的 PDF 數量: {len(existing_pdfs)} 個")
    for pdf in existing_pdfs:
        print(f"     * 檔名: {pdf.name} (大小: {pdf.stat().st_size / 1024 / 1024:.2f} MB)")
        
    if not target_test_pdf.exists():
        print(f"⚠️ 警告: 找不到預設測試論文 `{target_test_pdf.name}`")
        print(f"   這將導致 tests/ 內的自動化測試 (如 test_pdf_parser.py 等) 無法直接運行。")
        print(f"   👉 建議: 請上傳或將 'Attention Is All You Need' 的 PDF 檔案改名為")
        print(f"            'Attentation Is All You Need.pdf' 並放入 {DATA_DIR} 資料夾下。")
    else:
        print(f"✅ 成功找到預設測試論文: {target_test_pdf.name}")

    print("\n==================================================")
    print(" 💡 本地運行與測試指南指令")
    print("==================================================")
    print("1. 啟動 Streamlit 主應用程式 (已改為由 UI 介面輸入金鑰):")
    print("   👉 uv run python -m streamlit run app.py")
    print("\n2. 執行基礎環境與模組匯入測試:")
    print("   👉 uv run python tests/test_imports.py")
    print("\n3. 測試 Gemini Embedding 連線與向量轉換 (手動代入環境變數):")
    print("   👉 PowerShell: $env:GEMINI_API_KEY=\"您的金鑰\"; uv run python tests/test_gemini_emb.py")
    print("   👉 Bash/zsh: GEMINI_API_KEY=\"您的金鑰\" uv run python tests/test_gemini_emb.py")
    print("\n4. 測試 PDF 雙欄解析功能 (需先放 Attention 論文至 data/):")
    print("   👉 uv run python tests/test_pdf_parser.py")
    print("\n5. 測試向量庫寫入與語意檢索 (手動代入環境變數且放好論文):")
    print("   👉 PowerShell: $env:GEMINI_API_KEY=\"您的金鑰\"; uv run python tests/test_vectorstore.py")
    print("   👉 Bash/zsh: GEMINI_API_KEY=\"您的金鑰\" uv run python tests/test_vectorstore.py")
    print("==================================================")

if __name__ == "__main__":
    run_diagnostics()

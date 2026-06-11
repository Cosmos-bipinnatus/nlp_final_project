# -*- coding: utf-8 -*-
"""
一鍵測試批次執行器 (run_all_tests.py)
用於自動發現並依序執行 tests/ 目錄下的所有測試腳本，
並提供結構化的測試結果摘要。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# 確保輸出支援 UTF-8，避免 Windows 亂碼或崩潰
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_test_files(tests_dir: Path) -> List[Path]:
    """獲取 tests/ 目錄下所有以 test_ 開頭的 Python 檔案"""
    test_files: List[Path] = []
    # 遍歷 tests 目錄，排除 run_all_tests.py 本身以及 check_paths.py
    for file in tests_dir.glob("test_*.py"):
        if file.name != "test_imports.py": # 將 imports 移到最前面優先測試
            test_files.append(file)
    # 排序確保執行順序一致，並將 test_imports 放在第一個執行
    test_files.sort(key=lambda x: x.name)
    return [tests_dir / "test_imports.py"] + test_files

def run_single_test(test_file: Path) -> bool:
    """執行單個測試檔案並返回其成功狀態"""
    print(f"\n🚀 正在執行測試: {test_file.name} ...")
    print("-" * 50)
    
    try:
        # 使用當前 Python 虛擬環境執行
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=False, # 直接輸出到終端機以利即時觀察
            text=True,
            encoding="utf-8"
        )
        success = result.returncode == 0
        if success:
            print(f"✅ {test_file.name} 測試通過")
        else:
            print(f"❌ {test_file.name} 測試失敗 (Exit Code: {result.returncode})")
        return success
    except Exception as e:
        print(f"❌ 執行測試 `{test_file.name}` 時發生錯誤: {e}")
        return False

def main() -> None:
    """主控執行邏輯"""
    tests_dir = Path(__file__).resolve().parent
    test_files = get_test_files(tests_dir)
    
    print("==================================================")
    print(" 🛠️  Literature Reviewer 測試套件批次執行器")
    print("==================================================")
    
    # 若環境變數沒有 GEMINI_API_KEY，則提示使用者直接在終端機輸入
    if not os.getenv("GEMINI_API_KEY"):
        print("🔑 未偵測到 GEMINI_API_KEY 環境變數。")
        try:
            user_key = input("👉 請直接在此輸入您的 Gemini API 金鑰 (直接 Enter 則跳過 API 測試): ").strip()
            if user_key:
                os.environ["GEMINI_API_KEY"] = user_key
                print("✅ 已於當前測試工作階段暫時設定 GEMINI_API_KEY 環境變數。")
            else:
                print("ℹ️ 未輸入金鑰，後續需要 API 的測試項目將會被自動跳過。")
        except Exception:
            print("ℹ️ 無法讀取輸入，後續需要 API 的測試項目將會被自動跳過。")
            
    print(f"\n找到 {len(test_files)} 個測試檔案。將依序執行...")
    
    results: Dict[str, str] = {}
    
    for test_file in test_files:
        # 由於 test_pdf_parser 等需要 data/ 資料夾下的 PDF，若缺少則跳過
        if test_file.name in ["test_pdf_parser.py", "test_vectorstore.py", "test_pipeline.py", "test_comparison.py", "test_agent.py"]:
            data_dir = tests_dir.parent / "data"
            pdf_file = data_dir / "Attentation Is All You Need.pdf"
            if not pdf_file.exists():
                print(f"\n⏭️  跳過 {test_file.name} (原因: 缺少測試 PDF 數據 '{pdf_file.name}')")
                results[test_file.name] = "SKIP (Missing PDF)"
                continue
                
        # 由於需要 API 金鑰的測試，若沒有 .env 則跳過
        if test_file.name in ["test_gemini_emb.py", "test_vectorstore.py", "test_pipeline.py", "test_comparison.py", "test_agent.py"]:
            if not os.getenv("GEMINI_API_KEY"):
                print(f"\n⏭️  跳過 {test_file.name} (原因: 缺少 GEMINI_API_KEY 環境變數)")
                results[test_file.name] = "SKIP (Missing API Key)"
                continue
        
        success = run_single_test(test_file)
        results[test_file.name] = "PASSED" if success else "FAILED"
        
    print("\n==================================================")
    print(" 📊 測試結果摘要 (Summary)")
    print("==================================================")
    for name, status in results.items():
        status_color = "🟢" if status == "PASSED" else "🟡" if "SKIP" in status else "🔴"
        print(f"{status_color} {name:<25}: {status}")
    print("==================================================")

if __name__ == "__main__":
    main()

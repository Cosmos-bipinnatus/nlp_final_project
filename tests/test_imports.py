# -*- coding: utf-8 -*-
"""
測試模組匯入與設定載入。
"""
import sys
from pathlib import Path

# 將專案根目錄加入路徑
sys.path.append(str(Path(__file__).parent.parent))

# 避免 Windows 主機 console 輸出不支援的 utf-8 字元而崩潰
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_all_imports():
    print("=== 開始測試模組匯入 ===")
    
    try:
        # 1. 測試設定模組匯入
        from src.config.settings import (
            BASE_DIR,
            DATA_DIR,
            DOCS_DIR,
            VECTORSTORE_DIR,
            MODEL_NAME,
            EMBEDDING_MODEL,
        )
        print("✅ 成功匯入 src.config.settings")
        print(f"   BASE_DIR: {BASE_DIR}")
        print(f"   DATA_DIR: {DATA_DIR}")
        print(f"   VECTORSTORE_DIR: {VECTORSTORE_DIR}")
        print(f"   MODEL_NAME: {MODEL_NAME}")
        print(f"   EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        
        # 2. 測試日誌系統
        from src.utils.logger import get_logger
        logger = get_logger("test_imports")
        print("✅ 成功匯入 src.utils.logger")
        logger.info("日誌測試成功！已成功經由 get_logger 輸出日誌訊息。")
        
        # 3. 測試其餘模組匯入
        from src.loaders.pdf_parser import DoubleColumnPDFParser
        print("✅ 成功匯入 src.loaders.pdf_parser.DoubleColumnPDFParser")
        
        from src.rag.text_splitter import AcademicTextSplitter
        print("✅ 成功匯入 src.rag.text_splitter.AcademicTextSplitter")
        
        from src.rag.vector_manager import AcademicVectorManager
        print("✅ 成功匯入 src.rag.vector_manager.AcademicVectorManager")
        
        from src.rag.generator import AcademicRAGGenerator
        print("✅ 成功匯入 src.rag.generator.AcademicRAGGenerator")
        
        from src.rag.comparison_manager import AcademicComparisonManager
        print("✅ 成功匯入 src.rag.comparison_manager.AcademicComparisonManager")
        
        from src.agents.router_agent import AcademicRouterAgent
        print("✅ 成功匯入 src.agents.router_agent.AcademicRouterAgent")
        
        from src.tools.local_search_tool import LocalSearchTool
        from src.tools.arxiv_search_tool import ArXivSearchTool
        print("✅ 成功匯入 src.tools.local_search_tool & ArXivSearchTool")
        
        print("\n🎉 所有模組匯入與全域設定測試成功！")
        
    except Exception as e:
        print(f"\n❌ 測試失敗，發生異常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_all_imports()

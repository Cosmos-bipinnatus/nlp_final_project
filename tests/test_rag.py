# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# 將 src/ 目錄加入 Python 模組搜尋路徑，確保可以順利 import src 中的模組
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.vector_manager import AcademicVectorManager
from src.rag.generator import AcademicRAGGenerator

def test_rag_pipeline():
    print("[Info] Starting RAG Pipeline and Citation Verification Test...")
    
    try:
        # 1. 初始化向量管理器與 RAG 生成器
        print("\n--- Step 1: Connecting to ChromaDB & Gemini Chat Model ---")
        vector_manager = AcademicVectorManager(persist_directory="vectorstore")
        generator = AcademicRAGGenerator()
        
        # 確認資料庫中是否有數據
        db_count = vector_manager.get_collection_count()
        print(f"Current DB Count: {db_count} chunks.")
        
        if db_count == 0:
            print("[Warning] Database is empty! Please run tests/test_vectorstore.py first to ingest classic papers.")
            return
            
        # 2. 定義提問
        test_query = "What is the significance of Multi-Head Attention in the Transformer?"
        print(f"\nStudent Query: '{test_query}'")
        
        # 3. 執行相似度檢索 (Retriever)
        print("\n--- Step 2: Retrieving Top 3 Context Chunks from ChromaDB ---")
        retrieved_chunks = vector_manager.semantic_search(test_query, k=3)
        print(f"[Success] Retrieved {len(retrieved_chunks)} chunks.")
        
        # 4. 呼叫 RAG 生成回答 (Generator)
        print("\n--- Step 3: Generating Answer with Sentence-Level Citations ---")
        cited_answer = generator.generate_answer(test_query, retrieved_chunks)
        
        # 5. 輸出結果
        print("\n" + "="*80)
        print("[Results] RAG Cited Answer:")
        print("="*80)
        
        # 處理 Windows cmd/powershell cp950 編碼問題
        sys.stdout.reconfigure(errors='ignore') if hasattr(sys.stdout, 'reconfigure') else None
        
        print(cited_answer)
        print("="*80)
        print("\n✅ Week 4 RAG Generator and Citation Tags Pipeline Test Completed Successfully!")
        
    except Exception as e:
        print(f"[Error] RAG pipeline test failed: {e}")

if __name__ == "__main__":
    test_rag_pipeline()

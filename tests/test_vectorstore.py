import sys
import os
from pathlib import Path

# 將 src/ 目率加入 Python 模組搜尋路徑，確保可以順利 import src 中的模組
sys.path.append(str(Path(__file__).parent.parent))

from src.loaders.pdf_parser import DoubleColumnPDFParser
from src.rag.text_splitter import AcademicTextSplitter
from src.rag.vector_manager import AcademicVectorManager

def test_vectorstore_pipeline():
    pdf_path = Path("data") / "Attentation Is All You Need.pdf"
    
    if not pdf_path.exists():
        print(f"[Error] Can not find PDF: {pdf_path}")
        return
        
    print("[Info] Starting Vector Store Ingest and Semantic Search Test...")
    
    try:
        # 1. 執行雙欄解析管線
        print("\n--- Step 1: Parsing PDF ---")
        parser = DoubleColumnPDFParser(pdf_path)
        parsed_pages = parser.parse_pdf()
        print(f"[Success] Parsed {len(parsed_pages)} pages.")
        
        # 2. 執行文本切塊管線
        print("\n--- Step 2: Chunking Text ---")
        splitter = AcademicTextSplitter(chunk_size=600, chunk_overlap=120)
        chunks = splitter.split_parsed_documents(parsed_pages)
        print(f"[Success] Generated {len(chunks)} chunks.")
        
        # 3. 初始化本地向量資料庫與 Gemini Embedding
        print("\n--- Step 3: Connecting to ChromaDB with Gemini Embeddings ---")
        vector_manager = AcademicVectorManager(persist_directory="vectorstore")
        
        # 4. 重置資料庫，確保測試結果不被舊資料干擾
        print("\n--- Step 4: Resetting Local Database ---")
        vector_manager.clear_database()
        print(f"Current DB Count: {vector_manager.get_collection_count()} chunks.")
        
        # 5. 向量化並儲存文獻切塊
        print("\n--- Step 5: Vectorizing and Indexing Chunks (Uploading to Gemini Embeddings) ---")
        # 這會調用 Gemini text-embedding-004 API，需要點時間
        vector_manager.store_documents(chunks)
        print(f"[Success] Persistent store complete. DB Count: {vector_manager.get_collection_count()} chunks.")
        
        # 6. 語意相似度檢索測試 (Recall Test)
        print("\n--- Step 6: Performing Semantic Search Recall Test ---")
        test_query = "What is the significance of Multi-Head Attention in the Transformer?"
        print(f"Search Query: '{test_query}'")
        
        # 取得最相似的 3 個切塊
        search_results = vector_manager.semantic_search(test_query, k=3)
        
        print("\n" + "="*70)
        print("[Results] Semantic Search Recall Results (Top 3):")
        print("="*70)
        
        # 處理 cp950 編碼問題
        sys.stdout.reconfigure(errors='ignore') if hasattr(sys.stdout, 'reconfigure') else None
        
        for idx, (doc, score) in enumerate(search_results, 1):
            print(f"\n[Rank {idx}] | L2 Distance Score: {score:.4f} (smaller is closer)")
            print(f"Source: {doc.metadata['source']} | Page: {doc.metadata['page']} / {doc.metadata['total_pages']}")
            print(f"Chunk Index: {doc.metadata['chunk_index']}")
            print("-" * 50)
            print(doc.page_content.strip()[:350] + "\n...")
            print("-" * 50)
            
        print("\n✅ Week 3 ChromaDB + Gemini Embedding Pipeline Test Completed Successfully!")
        
    except Exception as e:
        print(f"[Error] Vector pipeline test failed: {e}")

if __name__ == "__main__":
    test_vectorstore_pipeline()

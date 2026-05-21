import sys
from pathlib import Path

# 將 src/ 目錄加入 Python 模組搜尋路徑，確保可以順利 import src 中的模組
sys.path.append(str(Path(__file__).parent.parent))

from src.loaders.pdf_parser import DoubleColumnPDFParser
from src.rag.text_splitter import AcademicTextSplitter

def test_complete_pipeline():
    pdf_path = Path("data") / "Attentation Is All You Need.pdf"
    
    if not pdf_path.exists():
        print(f"[Error] Can not find PDF: {pdf_path}")
        return
        
    print(f"[Info] Starting Ingest Pipeline Integration Test on: {pdf_path.name}")
    
    try:
        # 1. 執行雙欄解析管線
        print("\n--- Step 1: Parsing Double-Column PDF ---")
        parser = DoubleColumnPDFParser(pdf_path)
        parsed_pages = parser.parse_pdf()
        print(f"[Success] Parsed {len(parsed_pages)} pages.")
        
        # 2. 執行文本切塊管線
        print("\n--- Step 2: Splitting Text into Chunks ---")
        splitter = AcademicTextSplitter(chunk_size=600, chunk_overlap=120)
        chunks = splitter.split_parsed_documents(parsed_pages)
        print(f"[Success] Generated {len(chunks)} chunks.")
        
        # 3. 輸出分析結果
        print("\n" + "="*60)
        print("[Analysis] Pipeline Integration Statistics:")
        print(f"Total Pages: {len(parsed_pages)}")
        print(f"Total Chunks: {len(chunks)}")
        
        # 計算平均長度
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
        print(f"Average Chunk Length: {avg_len:.1f} characters")
        print("="*60)
        
        # 4. 列印前 3 個切塊範例，展示 Metadata 繼承狀態
        print("\n[Sample] === Displaying First 3 Chunks with Metadata ===")
        # 避免 cp950 編碼問題
        sys.stdout.reconfigure(errors='ignore') if hasattr(sys.stdout, 'reconfigure') else None
        
        for idx in range(min(3, len(chunks))):
            chunk = chunks[idx]
            print(f"\n[Chunk {chunk.metadata['chunk_index']}]")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Page Number: {chunk.metadata['page']} / {chunk.metadata['total_pages']}")
            print(f"Content length: {len(chunk.page_content)} chars")
            print("-" * 30)
            print(repr(chunk.page_content[:200]) + "...")
            print("-" * 30)
            
        print("\n✅ Ingest Pipeline (Parser + Splitter) Integration Test Successful!")
        
    except Exception as e:
        print(f"[Error] Pipeline test failed: {e}")

if __name__ == "__main__":
    test_complete_pipeline()

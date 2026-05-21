import sys
from pathlib import Path

# 將 src/ 目錄加入 Python 模組搜尋路徑，確保可以順利 import src 中的模組
sys.path.append(str(Path(__file__).parent.parent))

# pyrefly: ignore [missing-import]
from src.loaders.pdf_parser import DoubleColumnPDFParser

def test_parsing():
    # 尋找使用者上傳的 Attention 論文
    pdf_path = Path("data") / "Attentation Is All You Need.pdf"
    
    if not pdf_path.exists():
        print(f"[Error] Can not find PDF: {pdf_path}")
        print("Please check if the file is in the data/ folder and the name is spelled correctly.")
        return
        
    print(f"[Info] Detected test paper: {pdf_path.name}")
    print("[Info] Starting double-column sorting and parsing pipeline...")
    
    try:
        parser = DoubleColumnPDFParser(pdf_path)
        documents = parser.parse_pdf()
        
        print("\n" + "="*50)
        print("[Success] Parsing validation completed successfully!")
        print(f"Paper Filename: {documents[0]['metadata']['source']}")
        print(f"Total Pages: {documents[0]['metadata']['total_pages']}")
        print("="*50)
        
        # 列印第一頁（通常是 Title, Abstract 以及 Introduction 左欄）的解析範例
        print("\n[Sample] === Page 1 Text Segment (First 500 chars) ===")
        # 替換一些可能無法以 cp950 編碼顯示的特殊字元，避免 console 輸出崩潰
        content_p1 = documents[0]["page_content"][:500]
        # 在 Windows console 上安全列印，如果遇到編碼錯誤則忽略
        sys.stdout.reconfigure(errors='ignore') if hasattr(sys.stdout, 'reconfigure') else None
        print(content_p1)
        print("\n...")
        print("="*50)
        
        # 列印第二頁（通常開始進入標準雙欄）的解析範例
        if len(documents) > 1:
            print("\n[Sample] === Page 2 Text Segment (First 500 chars) ===")
            content_p2 = documents[1]["page_content"][:500]
            print(content_p2)
            print("\n...")
            print("="*50)
            
    except Exception as e:
        print(f"[Error] An error occurred during parsing: {e}")

if __name__ == "__main__":
    test_parsing()

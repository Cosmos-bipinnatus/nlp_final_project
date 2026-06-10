import os
import sys
from pathlib import Path

# 加入搜尋路徑
sys.path.append(str(Path(__file__).parent.parent))

from langchain_google_genai import GoogleGenerativeAIEmbeddings

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ 請提供 GEMINI_API_KEY 環境變數或輸入 API Key！")
        return
        
    print(f"正在測試 gemini-embedding-2-preview batch 轉換...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=api_key
        )
        # 測試輸入多個字串
        texts = ["Hello world", "This is a test of gemini embedding", "RAG systems are powerful"]
        result = embeddings.embed_documents(texts)
        print(f"✅ 成功！取得 {len(result)} 個 embedding 向量。")
        print(f"每個向量的維度: {len(result[0])}")
    except Exception as e:
        print(f"❌ 失敗！錯誤訊息: {e}")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

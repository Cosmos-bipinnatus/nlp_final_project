import os
import logging
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 載入環境變數
load_dotenv()

# 設定日誌記錄
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AcademicVectorManager:
    """
    學術論文向量資料庫管理員。
    負責使用 Google Gemini 的 text-embedding-004 模型將文本轉換為 768 維語意向量，
    並儲存於本地持久化的 ChromaDB 資料庫中，同時提供高效的相似度搜尋介面。
    """
    
    def __init__(self, persist_directory: str | Path = "vectorstore", collection_name: str = "literature_review"):
        """
        初始化向量資料庫管理員。
        
        Args:
            persist_directory (str | Path): ChromaDB 本地資料庫儲存路徑
            collection_name (str): 向量集合名稱（資料庫中的資料表名）
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # 1. 取得並檢查 Gemini API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.critical("未在環境變數中設定有效的 GEMINI_API_KEY！")
            raise ValueError("GEMINI_API_KEY 未正確載入，請檢查根目錄的 .env 檔案。")
            
        # 2. 初始化 Google Gemini Embeddings 模型
        # 指定 models/gemini-embedding-2 作為底層向量化模型（輸出 3072 維語意向量）
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            logger.info("成功初始化 Google Gemini Embeddings 模型 (gemini-embedding-001)。")
        except Exception as e:
            logger.error(f"初始化 Gemini Embeddings 時發生異常: {e}")
            raise e
            
        # 3. 初始化或載入本地 ChromaDB 向量庫
        try:
            self.vector_db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            logger.info(f"成功連結本地 ChromaDB 持久化目錄: {self.persist_directory}")
        except Exception as e:
            logger.error(f"初始化 ChromaDB 時發生異常: {e}")
            raise e

    def store_documents(self, documents: List[Document]) -> None:
        """
        將切塊後的 LangChain Document 轉換為向量並存入本地 ChromaDB 中。
        
        Args:
            documents (List[Document]): 待存入的切塊 Document 物件清單
        """
        if not documents:
            logger.warning("輸入的 Document 清單為空，跳過寫入。")
            return
            
        logger.info(f"準備向量化並儲存 {len(documents)} 個切塊到 ChromaDB...")
        try:
            # add_documents 會自動調用 embedding_function 對文本進行批次化向量轉換，並儲存至 SQLite+Parquet 本地儲存中
            self.vector_db.add_documents(documents)
            logger.info("ChromaDB 寫入成功！數據已持久化儲存。")
        except Exception as e:
            logger.error(f"ChromaDB 寫入失敗: {e}")
            raise e

    def semantic_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        進行語意相似度檢索。
        
        Args:
            query (str): 使用者的提問（如 'What is Multi-Head Attention?'）
            k (int): 預期召回 (Recall) 的最相似切塊數量
            
        Returns:
            List[Tuple[Document, float]]: 相似的 (Document, 距離分數) 元組清單。
            ChromaDB 的預設度量為 L2 距離 (Euclidean Distance)，
            分數越小代表向量距離越近，也就是語意相似度越高！
        """
        logger.info(f"開始進行語意檢索，查詢: '{query}' | 召回數 k={k}")
        try:
            # similarity_search_with_score 會回傳 (Document, L2_distance_score)
            results = self.vector_db.similarity_search_with_score(query, k=k)
            logger.info(f"語意檢索完成，共召回 {len(results)} 個切塊。")
            return results
        except Exception as e:
            logger.error(f"語意檢索過程中發生異常: {e}")
            raise e

    def get_collection_count(self) -> int:
        """
        查詢當前向量庫中的資料總筆數。
        """
        try:
            return self.vector_db._collection.count()
        except Exception as e:
            logger.error(f"查詢向量庫數量失敗: {e}")
            return 0

    def clear_database(self) -> None:
        """
        清空當前向量庫集合，用於重置文獻庫。
        """
        logger.warning(f"正在清空向量資料表: {self.collection_name}...")
        try:
            # 刪除集合並重新初始化以清空數據
            self.vector_db.delete_collection()
            self.vector_db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            logger.info("向量資料庫已成功清空。")
        except Exception as e:
            logger.error(f"清空資料庫時發生錯誤: {e}")
            raise e

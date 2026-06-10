import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.config import VECTORSTORE_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from src.utils.logger import get_logger

# 載入環境變數
load_dotenv()

# 設定日誌記錄
logger = get_logger(__name__)

class AcademicVectorManager:
    """
    學術論文向量資料庫管理員。
    負責使用 Google Gemini 的向量化模型將文本轉換為語意向量，
    並儲存於本地持久化的 ChromaDB 資料庫中，同時提供高效的相似度搜尋介面。
    """
    
    def __init__(self, persist_directory: str | Path = VECTORSTORE_DIR, collection_name: str = COLLECTION_NAME):
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
        # 指定底層向量化模型（當前使用 models/gemini-embedding-001，輸出 768 維語意向量）
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=api_key
            )
            logger.info(f"成功初始化 Google Gemini Embeddings 模型 ({EMBEDDING_MODEL})。")
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
            # 如果發生維度不相容等錯誤，自動清空並重建資料庫以自我修復 (B1 功能)
            err_msg = str(e)
            if any(word in err_msg.lower() for word in ["dimension", "dimensionality", "metadata", "invalid"]):
                logger.warning(
                    f"⚠️ 偵測到向量庫維度不相容或配置不匹配 ({err_msg})。"
                    f"系統將自動重置持久化目錄 `{self.persist_directory}` 並重建資料庫..."
                )
                try:
                    import shutil
                    if self.persist_directory.exists():
                        shutil.rmtree(self.persist_directory)
                    self.persist_directory.mkdir(parents=True, exist_ok=True)
                    
                    self.vector_db = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=str(self.persist_directory)
                    )
                    logger.info("ChromaDB 已成功清空並重新初始化。")
                except Exception as rebuild_error:
                    logger.critical(f"自動重建 ChromaDB 失敗: {rebuild_error}")
                    raise rebuild_error
            else:
                logger.error(f"初始化 ChromaDB 時發生異常: {e}")
                raise e

    def store_documents(self, documents: List[Document]) -> None:
        """
        將切塊後的 LangChain Document 轉換為向量並存入本地 ChromaDB 中。
        採用分批寫入 (Batching) 與指數退避重試機制，以避免 Gemini Embeddings API 的 429 速率限制錯誤。
        本方法具備「重複文件偵測與覆蓋」功能 (C3)，若偵測到相同來源的文獻已存在，會自動先清空舊的切塊再寫入。
        
        Args:
            documents (List[Document]): 待存入的切塊 Document 物件清單
        """
        if not documents:
            logger.warning("輸入的 Document 清單為空，跳過寫入。")
            return
            
        logger.info(f"準備向量化並儲存 {len(documents)} 個切塊到 ChromaDB...")
        
        # 1. 重複文件偵測與覆蓋處理 (C3)
        try:
            # 取得待寫入文獻的所有不重複來源檔名
            sources_to_write = {doc.metadata.get("source") for doc in documents if doc.metadata.get("source")}
            if sources_to_write:
                existing_sources = set(self.get_unique_sources())
                for source in sources_to_write:
                    if source in existing_sources:
                        logger.info(f"[重複偵測] 發現文獻 `{source}` 已存在於向量資料庫中，將進行覆蓋寫入（先刪除舊向量）。")
                        self.delete_by_source(source)
        except Exception as e:
            logger.warning(f"重複文件偵測/清理舊向量時發生非致命錯誤: {e}，將直接嘗試寫入。")
            
        from src.utils.retry_handler import retry_on_429
        
        # 設定每批次寫入的數量，避免單次 API 請求的 Payload 過大而觸發 rate limit
        batch_size = 15
        total_chunks = len(documents)
        
        try:
            for i in range(0, total_chunks, batch_size):
                batch = documents[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size
                
                logger.info(f"正在向量化寫入第 {batch_num}/{total_batches} 批，共 {len(batch)} 個切塊...")
                
                # 使用 retry_on_429 包裝 add_documents 呼叫，若遇到限流會自動等待重試
                retry_on_429(self.vector_db.add_documents, batch)
                
            logger.info("ChromaDB 所有批次寫入成功！數據已持久化儲存。")
        except Exception as e:
            logger.error(f"ChromaDB 批次寫入失敗: {e}")
            raise e

    def semantic_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        進行語意相似度檢索。
        採用指數退避重試機制，防止檢索時因高頻率發送 Embedding 請求而觸發 429 限流。
        
        Args:
            query (str): 使用者的提問（如 'What is Multi-Head Attention?'）
            k (int): 預期召回 (Recall) 的最相似切塊數量
            
        Returns:
            List[Tuple[Document, float]]: 相似的 (Document, 距離分數) 元組清單。
            ChromaDB 的預設度量為 L2 距離 (Euclidean Distance)，
            分數越小代表向量距離越近，也就是語意相似度越高！
        """
        logger.info(f"開始進行語意檢索，查詢: '{query}' | 召回數 k={k}")
        
        from src.utils.retry_handler import retry_on_429
        
        try:
            # similarity_search_with_score 會回傳 (Document, L2_distance_score)
            # 使用 retry_on_429 包裝檢索動作，因為檢索同樣會把 query 轉換為 embedding
            results = retry_on_429(self.vector_db.similarity_search_with_score, query, k=k)
            logger.info(f"語意檢索完成，共召回 {len(results)} 個切塊。")
            return results
        except Exception as e:
            logger.error(f"語意檢索過程中發生異常: {e}")
            raise e

    def delete_by_source(self, source_name: str) -> None:
        """
        刪除指定來源（例如某個 PDF 檔名）的所有向量資料切塊。
        
        Args:
            source_name (str): PDF 檔案名稱
        """
        logger.info(f"正在從向量庫中刪除來源為 `{source_name}` 的向量資料...")
        try:
            # ChromaDB 內建 collection 支援 where 條件刪除
            self.vector_db._collection.delete(where={"source": source_name})
            logger.info(f"成功刪除 `{source_name}` 的向量資料。")
        except Exception as e:
            logger.error(f"刪除 `{source_name}` 向量時發生錯誤: {e}")
            raise e

    def get_unique_sources(self) -> List[str]:
        """
        獲取向量庫中所有不重複的文獻來源（PDF 檔名）。
        
        Returns:
            List[str]: 不重複的 PDF 檔名清單
        """
        try:
            # 從 metadata 中獲取所有 source 欄位
            # ChromaDB python client _collection.get() 可以指定 include=['metadatas']
            results = self.vector_db._collection.get(include=['metadatas'])
            metadatas = results.get('metadatas', [])
            if not metadatas:
                return []
            
            sources = set()
            for meta in metadatas:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
            return list(sources)
        except Exception as e:
            logger.error(f"獲取向量庫來源清單失敗: {e}")
            return []

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

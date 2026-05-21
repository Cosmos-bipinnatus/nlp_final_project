# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any
from src.rag.vector_manager import AcademicVectorManager
from src.rag.generator import AcademicRAGGenerator

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LocalSearchTool:
    """
    本地文獻檢索與問答工具。
    串接本地的 ChromaDB 向量庫 (AcademicVectorManager) 與 RAG 生成器 (AcademicRAGGenerator)，
    針對已上傳並向量化的論文進行精確的學術檢索與句子級引用答案生成。
    """
    
    def __init__(self, persist_directory: str = "vectorstore"):
        """
        初始化本地搜尋工具。
        
        Args:
            persist_directory (str): 本地 ChromaDB 儲存路徑
        """
        logger.info("正在初始化本地文獻檢索工具...")
        try:
            self.vector_manager = AcademicVectorManager(persist_directory=persist_directory)
            self.generator = AcademicRAGGenerator()
            logger.info("本地文獻檢索工具初始化成功！")
        except Exception as e:
            logger.error(f"本地文獻檢索工具初始化失敗: {e}")
            raise e

    def run(self, query: str) -> Dict[str, Any]:
        """
        執行本地檢索與問答生成。
        
        Args:
            query (str): 學生的提問
            
        Returns:
            Dict[str, Any]: 包含 'answer' (帶引用回答) 與 'chunks' (召回的原始切塊) 的字典
        """
        logger.info(f"[LocalSearchTool] 開始處理查詢: '{query}'")
        try:
            # 1. 執行相似度檢索 (預設召回 Top 3 切塊)
            retrieved_chunks = self.vector_manager.semantic_search(query, k=3)
            logger.info(f"[LocalSearchTool] 語意搜尋召回完成，共召回 {len(retrieved_chunks)} 個切塊。")
            
            # 2. 調用 RAG 生成回答
            answer = self.generator.generate_answer(query, retrieved_chunks)
            
            return {
                "answer": answer,
                "chunks": retrieved_chunks
            }
        except Exception as e:
            logger.error(f"[LocalSearchTool] 執行失敗: {e}")
            return {
                "answer": f"❌ 本地檢索工具執行時發生錯誤: {e}",
                "chunks": []
            }

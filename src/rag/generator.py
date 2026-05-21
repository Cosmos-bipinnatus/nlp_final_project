# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts.academic_prompts import ACADEMIC_QA_SYSTEM_PROMPT, ACADEMIC_QA_USER_TEMPLATE

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AcademicRAGGenerator:
    """
    學術 RAG 生成器。
    基於 Google Gemini 2.5-Flash 模型，將召回的學術切塊（Context）與使用者提問（Query）
    結合成專門設計的學術提示詞，進行句子級別的精確引用標籤生成，確保學術嚴謹性並杜絕幻覺。
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
        """
        初始化學術 RAG 生成器。
        
        Args:
            model_name (str): 使用的 Gemini 模型名稱
            temperature (float): 溫度參數。設為較低值（如 0.2）可確保回答的高度嚴謹、高度基於事實，防止虛構。
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # 1. 取得並檢查 Gemini API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.critical("未在環境變數中設定有效的 GEMINI_API_KEY！")
            raise ValueError("GEMINI_API_KEY 未正確載入，請檢查根目錄的 .env 檔案。")
            
        # 2. 初始化 Gemini 聊天模型
        try:
            self.chat = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature,
                max_tokens=2048
            )
            logger.info(f"成功初始化 ChatGoogleGenerativeAI，模型: {self.model_name} (Temp={self.temperature})")
        except Exception as e:
            logger.error(f"初始化 ChatGoogleGenerativeAI 失敗: {e}")
            raise e

    def build_context_string(self, retrieved_chunks: List[Tuple[Document, float]]) -> str:
        """
        將召回的 (Document, score) 元組列表序列化為精美的 Context 字串。
        
        Args:
            retrieved_chunks (List[Tuple[Document, float]]): 召回的最相關切塊與 L2 距離元組清單。
            
        Returns:
            str: 格式化的 Context 文字區塊。
        """
        if not retrieved_chunks:
            return "（目前向量庫中無相關參考文獻切塊）"
            
        context_blocks = []
        for idx, (doc, score) in enumerate(retrieved_chunks, 1):
            source = doc.metadata.get("source", "未知文獻")
            page = doc.metadata.get("page", "?")
            total_pages = doc.metadata.get("total_pages", "?")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            
            # 組裝每一個切塊的結構化標頭與內容
            block = (
                f"--- [文獻 {idx}] 來源: {source} | 頁碼: p.{page} / {total_pages} | L2相似距: {score:.4f} | 切塊索引: #{chunk_idx} ---\n"
                f"{doc.page_content.strip()}"
            )
            context_blocks.append(block)
            
        return "\n\n".join(context_blocks)

    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[Document, float]]) -> str:
        """
        將檢索到的文獻切塊與問題組裝，呼叫 Gemini API 生成帶有精確引用標記的答案。
        
        Args:
            query (str): 使用者的提問（如 'What is Multi-Head Attention?'）
            retrieved_chunks (List[Tuple[Document, float]]): 由向量庫檢索出來的高相關文獻切塊與分數
            
        Returns:
            str: 帶有學術引用標記的回答。
        """
        # 防呆：如果沒有召回任何切塊，直接給予誠實的回絕，拒絕無事實依據的生成
        if not retrieved_chunks:
            logger.warning("沒有召回任何相關文獻切塊，拒絕生成以免幻覺。")
            return (
                "抱歉，根據目前已向量化的文獻庫，未發現與您的問題直接相關的研究數據或學術結論。\n"
                "💡 **建議**：請先前往「上傳文獻區」上傳包含該主題的 PDF 論文，並點擊「向量化本地文獻庫」以利系統檢索。"
            )
            
        # 1. 序列化 Context
        context_str = self.build_context_string(retrieved_chunks)
        
        # 2. 格式化使用者 Prompt 範本
        user_content = ACADEMIC_QA_USER_TEMPLATE.format(
            context_str=context_str,
            query=query
        )
        
        # 3. 包裝成 LangChain Core Messages
        messages = [
            SystemMessage(content=ACADEMIC_QA_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ]
        
        logger.info("開始呼叫 Gemini API 進行學術 RAG 回答生成...")
        try:
            # 4. 呼叫模型
            response = self.chat.invoke(messages)
            logger.info("Gemini API 回答生成完成！")
            return response.content
        except Exception as e:
            logger.error(f"Gemini API 呼叫失敗: {e}")
            raise e

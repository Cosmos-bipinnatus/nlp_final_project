# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag.vector_manager import AcademicVectorManager

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PaperFeatures(BaseModel):
    """
    學術論文核心特徵 Pydantic 結構化格式。
    """
    title: str = Field(description="論文名稱 (Title)。如果是 PDF 檔名，請萃取其代表的實際論文名稱（例如 'Attention Is All You Need'）。")
    methodology: str = Field(description="該論文提出的核心方法、演算法或架構說明（必須使用繁體中文，約 80-150 字），簡述其如何解決所面臨的挑戰。")
    datasets: str = Field(description="論文實驗中使用的資料集 (Datasets) 以及採用的評估指標 (Metrics，例如 BLEU、ROUGE、Accuracy 等)。必須使用繁體中文。")
    pros: str = Field(description="該論文的主要貢獻、優點與創新之處（例如計算效率提升、解決了長期依賴問題等）。必須使用繁體中文。")
    cons: str = Field(description="該論文提及的局限性、缺點或未來研究方向（例如對長文本處理效能下降、需要高運算成本等）。必須使用繁體中文。")

class AcademicComparisonManager:
    """
    跨文獻學術比較矩陣管理員。
    利用主題式 RAG 檢索與 Pydantic 結構化特徵提取技術，
    對本地上傳的多篇學術 PDF 進行交叉比對並生成對照表格。
    """
    
    def __init__(self, persist_directory: str = "vectorstore"):
        """
        初始化比較管理員。
        """
        self.persist_directory = persist_directory
        self.vector_manager = AcademicVectorManager(persist_directory=self.persist_directory)
        
        # 載入 API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 未設定，請檢查環境變數。")
            
        # 初始化 Gemini Pro LLM，使用免費高效且支援結構化輸出的 gemini-2.5-flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2, # 使用低溫度降低幻覺，確保內容客觀
            google_api_key=api_key
        )
        
        # 綁定結構化輸出格式
        self.structured_llm = self.llm.with_structured_output(PaperFeatures)
        logger.info("成功初始化 AcademicComparisonManager 暨 Pydantic 結構化特徵提取引擎。")

    def retrieve_paper_context(self, pdf_name: str, k: int = 6) -> str:
        """
        針對特定論文 PDF，以特徵萃取為主題，進行限定範圍的語意搜尋，召回最相關的 chunks 作為 Context。
        """
        # 主題式檢索查詢 (Topic-Specific Retrieval Query)
        theme_query = (
            "methodology proposed approach model architecture datasets "
            "experiments metrics evaluation results strengths weaknesses limitations pros cons"
        )
        
        logger.info(f"正在對文獻進行限定主題檢索: {pdf_name} | 檢索關鍵字: '{theme_query}'")
        
        try:
            # 透過 Chroma 進行 source 限定過濾 (Chroma 支援 filter)
            # 在 langchain_chroma 中，filter 參數是作為 kwargs 傳遞給 similarity_search
            # 格式為 filter={"source": pdf_name}
            results = self.vector_manager.vector_db.similarity_search_with_score(
                query=theme_query,
                k=k,
                filter={"source": pdf_name}
            )
            
            if not results:
                logger.warning(f"對文獻 {pdf_name} 的限定檢索結果為空，改用全局相似度搜尋兜底。")
                # 如果沒有找到，嘗試不要加 filter
                results = self.vector_manager.vector_db.similarity_search_with_score(
                    query=theme_query,
                    k=k
                )
            
            # 彙整內容
            context_pieces = []
            for idx, (doc, score) in enumerate(results, 1):
                page_info = doc.metadata.get("page", "?")
                context_pieces.append(f"[Source Piece #{idx} (Page {page_info})]\n{doc.page_content.strip()}")
                
            return "\n\n".join(context_pieces)
            
        except Exception as e:
            logger.error(f"提取文獻 {pdf_name} 的 RAG 上下文失敗: {e}")
            raise e

    def extract_features_for_paper(self, pdf_name: str) -> PaperFeatures:
        """
        對單篇論文 PDF 執行主題檢索並呼叫 Gemini 進行 Pydantic 結構化特徵提煉。
        """
        logger.info(f"開始為論文 '{pdf_name}' 提取學術特徵...")
        
        # 1. 召回特定主題 chunks 作為 Context
        context = self.retrieve_paper_context(pdf_name, k=8)
        
        if not context.strip():
            logger.warning(f"未能獲取論文 '{pdf_name}' 的任何 Context。")
            return PaperFeatures(
                title=pdf_name,
                methodology="無法檢索到相關文獻內容。",
                datasets="無法檢索到相關文獻內容。",
                pros="無法檢索到相關文獻內容。",
                cons="無法檢索到相關文獻內容。"
            )
            
        # 2. 設計特徵提取 Prompt
        prompt = (
            "你是一位資深的 AI/NLP 領域軟體工程師與學術導師。請根據下方提供的文獻上下文 (Context)，"
            f"精確提煉出該篇論文（原始檔名：{pdf_name}）的核心學術特徵。\n\n"
            "【限制要求】\n"
            "1. 必須嚴格根據提供的 Context 進行摘要，不得虛構或無中生有（防範學術幻覺）。\n"
            "2. 除了 Title 保留學術正式英文或中文外，其餘欄位（methodology, datasets, pros, cons）必須使用『臺灣地區學術常用繁體中文語境』進行撰寫。\n"
            "3. 核心方法 (methodology) 部分請摘要出論文提出的核心突破點，字數約 80-150 字。\n\n"
            f"【文獻上下文 Context】\n{context}\n\n"
            "請直接以結構化 JSON 輸出填寫 PaperFeatures 中的每個欄位。"
        )
        
        try:
            # 3. 呼叫 LLM 結構化輸出
            paper_features: PaperFeatures = self.structured_llm.invoke(prompt)
            logger.info(f"成功完成論文 '{pdf_name}' 的結構化特徵提煉: {paper_features.title}")
            return paper_features
        except Exception as e:
            logger.error(f"調用 LLM 結構化提煉論文 '{pdf_name}' 特徵失敗: {e}")
            raise e

    def generate_comparison_grid(self, pdf_names: List[str]) -> List[Dict[str, Any]]:
        """
        對多篇論文 PDF 進行批次特徵提取，並彙整為 Dict 串列以供前端轉為 DataFrame/Table 渲染。
        """
        if len(pdf_names) < 2:
            raise ValueError("至少需要選擇兩篇論文才能進行交叉比較矩陣的生成。")
            
        logger.info(f"開始生成跨文獻比較矩陣，共 {len(pdf_names)} 篇文獻: {pdf_names}")
        comparison_results = []
        
        for pdf_name in pdf_names:
            try:
                features = self.extract_features_for_paper(pdf_name)
                comparison_results.append({
                    "pdf_file": pdf_name,
                    "title": features.title,
                    "methodology": features.methodology,
                    "datasets": features.datasets,
                    "pros": features.pros,
                    "cons": features.cons
                })
            except Exception as e:
                logger.error(f"提取 '{pdf_name}' 失敗，跳過或以防呆數據兜底。")
                comparison_results.append({
                    "pdf_file": pdf_name,
                    "title": pdf_name,
                    "methodology": f"特徵提取失敗: {e}",
                    "datasets": "無資料",
                    "pros": "無資料",
                    "cons": "無資料"
                })
                
        return comparison_results

    @staticmethod
    def convert_to_markdown_table(comparison_data: List[Dict[str, Any]]) -> str:
        """
        將比較資料轉換為高可讀性的學術 Markdown 表格格式，供使用者下載複製。
        """
        md = []
        md.append("| 論文名稱 (Title) | 核心研究方法 (Methodology) | 實驗資料集與指標 (Datasets & Metrics) | 優點與創新 (Pros) | 局限性與缺點 (Cons) |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        
        for row in comparison_data:
            # 移除換行符號以避免破壞 Markdown 表格結構
            title = row['title'].replace('\n', ' ').replace('|', '\\|')
            meth = row['methodology'].replace('\n', ' ').replace('|', '\\|')
            data = row['datasets'].replace('\n', ' ').replace('|', '\\|')
            pros = row['pros'].replace('\n', ' ').replace('|', '\\|')
            cons = row['cons'].replace('\n', ' ').replace('|', '\\|')
            md.append(f"| **{title}**<br>_({row['pdf_file']})_ | {meth} | {data} | {pros} | {cons} |")
            
        return "\n".join(md)

# -*- coding: utf-8 -*-
import os
import re
from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import MODEL_NAME, VECTORSTORE_DIR
from src.utils.logger import get_logger

from src.tools.local_search_tool import LocalSearchTool
from src.tools.arxiv_search_tool import ArXivSearchTool

# 設定日誌
logger = get_logger(__name__)

# ==========================================
# 1. 定義路由決策的強型別資料結構 (Pydantic)
# ==========================================
class RouterDecision(BaseModel):
    """
    路由代理的決策格式。
    Gemini 將嚴格遵守此欄位約束進行 JSON 輸出。
    """
    chosen_route: Literal["local", "arxiv", "hybrid"] = Field(
        ..., 
        description="決定將此提問路由分發至何處。若問及本地上傳文獻的細節，選 'local'；若問及外部技術或搜尋最新論文，選 'arxiv'；若提問需要同時對比本地已上傳文獻與外部最新論文（例如詢問兩者之差異、最新進展等），選 'hybrid'。"
    )
    search_query: str = Field(
        ..., 
        description="為該檢索庫優化後的英文學術檢索關鍵字（例如：'Multi-Head Attention mechanism' 或 'diffusion model 3D generation'）"
    )
    rationale: str = Field(
        ..., 
        description="繁體中文決策理由。簡短且專業地向學生解釋：為什麼此提問應該被路由至該通道？"
    )

# ==========================================
# 2. 路由代理實作
# ==========================================
class AcademicRouterAgent:
    """
    學術路由代理。
    負責接收學生提問，決定最佳的資料檢索途徑，並調用相應工具執行。
    採用 Pydantic 結構化輸出模式，確保決策的透明度與防呆性。
    """
    
    def __init__(self, model_name: str = MODEL_NAME, persist_directory: str = str(VECTORSTORE_DIR)):
        """
        初始化路由代理，加載 Gemini LLM 並綁定結構化輸出，以及加載相關工具。
        
        Args:
            model_name (str): 用於決策的 Gemini 模型名稱
            persist_directory (str): 本地 ChromaDB 持久化路徑
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # 1. 初始化決策專用的 LLM (使用低 temperature 參數以確保決策高度穩定)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.critical("未在環境變數中設定有效的 GEMINI_API_KEY！")
            raise ValueError("GEMINI_API_KEY 未正確載入。")
            
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.1,  # 決策專用，越低越穩定
                max_tokens=1024
            )
            
            # 使用 LangChain Core 的 with_structured_output 綁定我們的 Pydantic schema
            self.structured_llm = self.llm.with_structured_output(RouterDecision)
            logger.info("成功載入 Router Agent LLM，並成功綁定 Pydantic 結構化輸出。")
        except Exception as e:
            logger.error(f"Router Agent LLM 綁定失敗: {e}")
            raise e
            
        # 2. 惰性加載 / 初始化底層工具
        self.local_tool = LocalSearchTool(persist_directory=self.persist_directory)
        self.arxiv_tool = ArXivSearchTool(model_name=self.model_name)

    def make_decision(self, query: str, chat_history: List[dict] = None) -> RouterDecision:
        """
        分析學生問題與對話歷史，進行語意分類並回傳結構化決策。
        
        Args:
            query (str): 學生的原始提問
            chat_history (List[dict]): 近期學術對話歷史
            
        Returns:
            RouterDecision: 路由決策模型執行個體
        """
        logger.info(f"[RouterAgent] 正在進行決策分析，問題為: '{query}'")
        
        system_prompt = (
            "你是一位資深的 AI / NLP 領域學術導師與智能路由引擎。\n"
            "你的唯一任務是分析學生提出的問題，並決定該問題應該路由分發至「本地文獻庫 (local)」、「外接 ArXiv 學術庫 (arxiv)」還是「兩者混合 (hybrid)」。\n\n"
            "【路由分發判定準則】\n"
            "1. 本地文獻庫 (local)：\n"
            "   - 提問涉及特定的細節內容，通常已經包含在學生已上傳的論文中（例如：Transformer、Attention 機制、自注意力、Multi-Head Attention 運作方式、或者特定的數據與研究方法）。\n"
            "   - 提問包含 '這篇論文'、'作者提出什麼'、'此研究'、'Attention' 等已上傳文獻的範圍。\n"
            "2. 外接 ArXiv 學術庫 (arxiv)：\n"
            "   - 提問是關於某個研究領域的最新論文（例如：'2025 年最新 RAG 研究'、'關於 AI 代理的最新文獻'）。\n"
            "   - 提問涉及本地未上傳之新領域或新話題（例如：Diffusion Model、LLM Agents、RLHF、DPO 等廣泛、非 Attention 原版論文範疇之學術主題）。\n"
            "   - 提問明確包含 '搜尋'、'查找最新'、'ArXiv 論文' 等字眼。\n"
            "3. 兩者混合 (hybrid)：\n"
            "   - 提問需要『將本地上傳文獻』與『外部最新進展』進行對照比較（例如：'我上傳的 Attention 論文與 2025 年最新 Attention 改進有什麼不同？'、'將本地文獻提出的方法與最新線上研究做對照' 等）。\n\n"
            "【英文關鍵字提取準則】\n"
            "無論選擇哪個路由，請為該提問提取並優化出 2~4 個關鍵字組合成的【英文學術檢索關鍵字】(search_query)，這對於後續在資料庫或 ArXiv 搜尋非常重要。切勿使用中文作為檢索字，必須轉化為英文學術專有名詞（例如：'Multi-head attention formulas' 或 'latest LLM evaluation frameworks'）。\n\n"
            "請嚴格回傳符合 RouterDecision 格式的結構化 JSON。"
        )
        
        # 將對話歷史融入 Prompt 以提供代理決策上下文
        history_str = ""
        if chat_history:
            history_str = "【近期學術對話歷史紀錄】\n"
            for msg in chat_history[-3:]: # 僅使用最近 3 輪對話以節省 Token
                role = "學生" if msg.get("role") == "user" else "AI導師"
                content = msg.get("plain_answer") or msg.get("content")
                # 簡單清理 HTML 標籤
                clean_content = re.sub(r'<[^>]+>', '', content)
                history_str += f"{role}: {clean_content}\n"
            history_str += "\n"
            
        user_prompt = f"{history_str}學生新提問：{query}\n\n請根據判定準則，做出你的路由決策與檢檢索關鍵字最佳化，並給出你的繁體中文決策理由："
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            from src.utils.retry_handler import retry_on_429
            decision = retry_on_429(self.structured_llm.invoke, messages)
            logger.info(
                f"[RouterAgent] 決策完成！路由選擇: '{decision.chosen_route}' | "
                f"英文檢索字: '{decision.search_query}' | "
                f"決策理由: {decision.rationale}"
            )
            return decision
        except Exception as e:
            logger.error(f"[RouterAgent] 決策過程中發生異常，回退至 local 路由: {e}")
            # 防呆回退機制
            return RouterDecision(
                chosen_route="local",
                search_query=query,
                rationale=f"因系統決策模組異常 ({e})，安全回退至本地檢索模式。"
            )

    def route_and_execute(self, query: str, chat_history: List[dict] = None) -> Dict[str, Any]:
        """
        進行路由決策，並直接分發、調用相應工具執行。
        
        Args:
            query (str): 學生的原始提問
            chat_history (List[dict]): 對話歷史紀錄
            
        Returns:
            Dict[str, Any]: 整合了代理決策歷程與工具執行結果的 JSON
        """
        # 1. 做出路由決策
        decision = self.make_decision(query, chat_history=chat_history)
        
        # 2. 根據決策分發執行
        try:
            if decision.chosen_route == "local":
                logger.info("[RouterAgent] 決策選擇本地 RAG，開始執行 LocalSearchTool...")
                tool_output = self.local_tool.run(query, chat_history=chat_history)
                return {
                    "route": "local",
                    "search_query": decision.search_query,
                    "rationale": decision.rationale,
                    "answer": tool_output["answer"],
                    "chunks": tool_output["chunks"],  # 用於前端來源展開檢視
                    "papers": []
                }
            elif decision.chosen_route == "arxiv":
                logger.info("[RouterAgent] 決策選擇外部搜尋，開始執行 ArXivSearchTool...")
                # 這裡使用最佳化後的英文學術檢索字來呼叫 ArXiv，效果最好！
                tool_output = self.arxiv_tool.run(decision.search_query)
                return {
                    "route": "arxiv",
                    "search_query": decision.search_query,
                    "rationale": decision.rationale,
                    "answer": tool_output["answer"],
                    "chunks": [],
                    "papers": tool_output["papers"]  # 用於前端論文卡片渲染
                }
            else:
                logger.info("[RouterAgent] 決策選擇混合路由 (hybrid)，同時查詢本地與 ArXiv...")
                local_output = self.local_tool.run(query, chat_history=chat_history)
                arxiv_output = self.arxiv_tool.run(decision.search_query)
                
                # 呼叫 Gemini LLM 進行學術級的內容融合
                logger.info("[RouterAgent] 正在使用 Gemini 進行本地 RAG 與 ArXiv 線上論文答案的混合融合...")
                system_prompt = (
                    "你是一位資深的 AI / NLP 領域學術導師與文獻綜述專家。\n"
                    "你的任務是將『本地已上傳文獻的檢索回答』與『ArXiv 線上最新的相關論文摘要綜述』進行學術級的對照綜述融合，\n"
                    "為學生提供一份既包含本地文獻細節，又包含線上最新進展的『混合對照綜述解答』。\n"
                    "請嚴格使用「中華民國繁體中文（臺灣地區學術用語）」進行撰寫。\n"
                )
                
                user_prompt = f"""學生的問題：{query}

【本地文獻檢索回答】：
{local_output["answer"]}

【ArXiv 線上檢索論文摘要綜述】：
{arxiv_output["answer"]}

請將上述兩部分的內容進行融合與交叉分析：
1. 說明本地文獻的研究與線上最新進展的關聯性或技術演進脈絡。
2. 點出兩者的技術差異、優勢與局限。
3. 產出一篇邏輯連貫、結構嚴密的學術解答。
4. 必須保留原本的所有引用標記（例如 `[論文A, p.4]` 或 `[1]`）。
"""
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                from src.utils.retry_handler import retry_on_429
                response = retry_on_429(self.llm.invoke, messages)
                
                return {
                    "route": "hybrid",
                    "search_query": decision.search_query,
                    "rationale": decision.rationale,
                    "answer": response.content,
                    "chunks": local_output["chunks"],
                    "papers": arxiv_output["papers"]
                }
        except Exception as e:
            logger.error(f"[RouterAgent] 工具執行時發生致命錯誤: {e}")
            return {
                "route": decision.chosen_route,
                "search_query": decision.search_query,
                "rationale": decision.rationale,
                "answer": f"❌ 系統在路由執行過程中發生錯誤: {e}",
                "chunks": [],
                "papers": []
            }

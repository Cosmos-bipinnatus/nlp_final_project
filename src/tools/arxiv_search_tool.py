# -*- coding: utf-8 -*-
import os
import arxiv
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import MODEL_NAME, ARXIV_MAX_RESULTS
from src.utils.logger import get_logger

# 設定日誌
logger = get_logger(__name__)

class ArXivSearchTool:
    """
    ArXiv 線上學術文獻搜尋工具。
    利用官方 arxiv API 檢索最新、最相關的學術論文，
    並結合 Google Gemini 2.5-Flash 模型進行「學術級繁體中文編譯與摘要整合」，
    幫助學生快速生成高品質的外部文獻回顧。
    """
    
    def __init__(self, model_name: str = MODEL_NAME, max_results: int = ARXIV_MAX_RESULTS):
        """
        初始化 ArXiv 搜尋工具。
        
        Args:
            model_name (str): 用於翻譯與摘要編譯的 Gemini 模型名稱
            max_results (int): 預設回傳的線上論文筆數，設為 3 筆以保證速度與版面美觀
        """
        self.model_name = model_name
        self.max_results = max_results
        
        # 初始化 ArXiv Client
        # 關鍵修正：設定 page_size 為實際需要的筆數（預設為 100 會觸發 HTTP 429 限流）
        # 同時設定 delay_seconds 為 5 秒，作為 ArXiv 伺服器的請求間緩衝
        self.client = arxiv.Client(
            page_size=max_results,
            delay_seconds=5.0,
            num_retries=3
        )
        
        # 初始化 Gemini 模型
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.critical("未在環境變數中設定有效的 GEMINI_API_KEY！")
            raise ValueError("GEMINI_API_KEY 未正確載入。")
            
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0.3,  # 稍微提高一點以利語言翻譯流暢性，但仍保持事實精準性
            max_tokens=2560
        )
        logger.info("ArXiv 搜尋工具初始化成功。")

    def _fetch_arxiv_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        底層方法：向 ArXiv API 發起查詢，抓取原始論文資料。
        """
        logger.info(f"正在向 ArXiv API 發起搜尋，查詢詞: '{query}'...")
        try:
            # 設計 Search 物件
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = list(self.client.results(search))
            logger.info(f"ArXiv 查詢完成，找到 {len(results)} 筆相關結果。")
            
            parsed_papers = []
            for paper in results:
                # 提取年份
                pub_year = paper.published.strftime("%Y")
                pub_date = paper.published.strftime("%Y-%m-%d")
                
                # 提取作者清單 (取前三位)
                authors_list = [author.name for author in paper.authors]
                if len(authors_list) > 3:
                    authors_str = ", ".join(authors_list[:3]) + " et al."
                else:
                    authors_str = ", ".join(authors_list)
                    
                parsed_papers.append({
                    "title": paper.title,
                    "authors": authors_str,
                    "published": pub_date,
                    "year": pub_year,
                    "summary": paper.summary.replace("\n", " ").strip(),
                    "pdf_url": paper.pdf_url,
                    "arxiv_id": paper.entry_id.split("/abs/")[-1] if "/abs/" in paper.entry_id else paper.entry_id
                })
            return parsed_papers
        except Exception as e:
            logger.error(f"ArXiv API 請求失敗: {e}")
            raise e

    def run(self, query: str) -> Dict[str, Any]:
        """
        執行 ArXiv 線上論文檢索與 Gemini 繁體中文摘要編譯。
        
        Args:
            query (str): 學術英文檢索關鍵字
            
        Returns:
            Dict[str, Any]: 包含 'answer' (Markdown 格式的中文文獻回顧報告) 與 'papers' (原始結構化數據) 的字典
        """
        logger.info(f"[ArXivSearchTool] 開始處理搜尋要求: '{query}'")
        try:
            # 1. 抓取 ArXiv 原始論文
            papers = self._fetch_arxiv_papers(query)
            
            if not papers:
                return {
                    "answer": f"🔍 系統在 ArXiv 學術庫中未檢索到與「{query}」相關的英文論文，建議調整您的關鍵字（盡量使用英文學術專有名詞）。",
                    "papers": []
                }
                
            # 2. 串接 Gemini 進行繁體中文翻譯與學術提煉
            logger.info("正在使用 Gemini 進行英文摘要的學術級繁體中文編譯...")
            
            # 組裝輸入文字
            arxiv_text_blocks = []
            for idx, p in enumerate(papers, 1):
                block = (
                    f"[論文 {idx}]\n"
                    f"Title: {p['title']}\n"
                    f"Authors: {p['authors']}\n"
                    f"Published: {p['published']}\n"
                    f"ArXiv Link: {p['pdf_url']}\n"
                    f"Abstract:\n{p['summary']}\n"
                )
                arxiv_text_blocks.append(block)
                
            arxiv_results_text = "\n\n".join(arxiv_text_blocks)
            
            # 設計 Prompt 系統提示
            system_prompt = (
                "你是一位資深的 AI / NLP 領域學術導師與文獻回顧專家。\n"
                "你將協助學生將 ArXiv 搜尋到的最新英文論文，編譯整理成一篇結構極其精美、易讀且具備高度學術深度的【繁體中文最新前沿論文文獻回顧】。\n"
                "請嚴格使用「中華民國繁體中文（臺灣地區學術用語）」進行撰寫（例如：資訊、切塊、注意力機制、向量、超參數、大型語言模型）。\n"
            )
            
            user_prompt = f"""以下是系統透過 ArXiv API 查詢到的前 {self.max_results} 筆最相關英文論文資料：

{arxiv_results_text}

請將這些英文論文資料，為大二學生進行「學術級繁體中文編譯與研究回顧整合」。
你的輸出必須採用 Markdown 格式，且必須嚴格包含以下結構：

1. **📈 前沿技術演進趨勢解析**：
   - 寫一段簡短的導言（150字以內），以宏觀的學術視角，分析這幾篇論文所揭示的最新技術走向與演變趨勢。

2. **📄 線上文獻回顧卡片組**：
   - 針對每篇論文，設計一個清晰的區塊，內容包含：
     - **論文名稱**（標題，中英對照，英文保持原樣）
     - **發表年份與作者**
     - **💡 核心亮點 (Core Contribution)**：用 1~2 句話，直擊痛點說明該研究最重要的技術貢獻或創新方法。
     - **🔬 技術細節與摘要編譯**：以流暢、易懂的繁體中文重新編譯並解析其 Abstract，說明其模型架構、提出的演算法與重要實驗數據成果。
     - **🔗 線上閱讀與 PDF 下載**：以 Markdown 連結方式提供該論文的 ArXiv PDF URL（例如 `[點此下載 ArXiv PDF 檔案](URL)`）。

請開始為學生進行這份精美的文獻回顧編譯："""

            # 呼叫 Gemini 2.5-Flash
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            from src.utils.retry_handler import retry_on_429
            response = retry_on_429(self.llm.invoke, messages)
            logger.info("Gemini 繁體中文文獻編譯完成！")
            
            return {
                "answer": response.content,
                "papers": papers
            }
            
        except Exception as e:
            logger.error(f"[ArXivSearchTool] 執行失敗: {e}")
            return {
                "answer": f"❌ ArXiv 線上檢索工具執行時發生錯誤: {e}",
                "papers": []
            }

# -*- coding: utf-8 -*-
"""
學術文獻回顧生成管理員
利用多篇論文的結構化特徵，結合 Gemini 生成高品質、語意連貫的學術文獻回顧報告。
"""

import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import MODEL_NAME
from src.utils.logger import get_logger

# 設定日誌
logger = get_logger(__name__)

class AcademicLiteratureReviewGenerator:
    """
    學術文獻回顧生成器。
    基於多篇已上傳論文的特徵資料（如研究方法、資料集、優缺點等），
    呼叫 Gemini 2.5-Flash 自動進行「交叉比對、技術演進分析與研究缺口提煉」，
    生成具備高度學術嚴謹性的文獻回顧報告。
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        初始化文獻回顧生成器。
        """
        self.model_name = model_name
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.critical("未在環境變數中設定有效的 GEMINI_API_KEY！")
            raise ValueError("GEMINI_API_KEY 未設定，請檢查環境變數。")
            
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.3,  # 適度調低溫度以保證學術論述的客觀嚴謹
                google_api_key=api_key,
                max_tokens=3072   # 增加 Token 上限以容納較長的文章生成
            )
            logger.info("成功初始化 AcademicLiteratureReviewGenerator。")
        except Exception as e:
            logger.error(f"初始化 ChatGoogleGenerativeAI 失敗: {e}")
            raise e

    def generate_review_report(self, comparison_data: List[Dict[str, Any]]) -> str:
        """
        根據多篇論文的特徵矩陣，生成結構化的 Markdown 文獻回顧報告。
        
        Args:
            comparison_data (List[Dict[str, Any]]): 來自 ComparisonManager 的論文特徵矩陣。
                包含 pdf_file, title, methodology, datasets, pros, cons 等欄位。
                
        Returns:
            str: Markdown 格式的文獻回顧報告。
        """
        if len(comparison_data) < 2:
            raise ValueError("至少需要兩篇論文才能生成文獻回顧報告。")
            
        logger.info(f"開始為 {len(comparison_data)} 篇文獻生成綜述報告...")
        
        # 1. 序列化特徵資料，作為 Context 輸入給 LLM
        serialized_features = []
        for idx, paper in enumerate(comparison_data, 1):
            block = (
                f"【文獻 #{idx}】\n"
                f"- 原始檔名: {paper.get('pdf_file')}\n"
                f"- 論文名稱 (Title): {paper.get('title')}\n"
                f"- 核心研究方法: {paper.get('methodology')}\n"
                f"- 實驗資料集與指標: {paper.get('datasets')}\n"
                f"- 主要優點與貢獻: {paper.get('pros')}\n"
                f"- 局限性與未來方向: {paper.get('cons')}\n"
            )
            serialized_features.append(block)
            
        features_context = "\n\n".join(serialized_features)
        
        # 2. 設計學術提示詞 (Academic Prompts)
        system_prompt = (
            "你是一位資深的 AI / NLP 領域學術導師與文獻綜述寫作專家。\n"
            "你的任務是根據學生提供的多篇論文之結構化特徵，撰寫一篇邏輯嚴密、結構完整、且具備高度學術價值的【多文獻交叉綜述報告】。\n"
            "請嚴格使用「中華民國繁體中文（臺灣地區學術用語）」進行撰寫（例如：資訊、自注意力機制、自然語言處理、大型語言模型）。\n"
        )
        
        user_prompt = f"""請根據以下已提取的多篇論文學術特徵：

{features_context}

為我撰寫一篇高品質的【學術文獻回顧報告】。
這篇報告是為大二學生準備的教學範本，旨在引導他們理解如何整合、比較多篇文獻。
你的輸出必須採用 Markdown 格式，且必須嚴格包含以下結構，不可缺少任何一個大項：

---

# 📚 多文獻學術綜述與研究回顧報告

## 1. 🎯 研究背景與綜述導言 (Introduction & Context)
   - 簡述這幾篇論文所屬的研究領域及其重要性。
   - 歸納這幾篇論文共同嘗試解決的核心科學痛點或挑戰。

## 2. 🔬 核心方法學對照分析 (Methodological Comparison)
   - 以客觀學術視角，深入比較各篇文獻所提出的解決方案與架構設計（例如各自採用了什麼特殊的 Attention 機制、特徵對齊等）。
   - 分析不同方法的適用情境。

## 3. 📈 技術演進脈絡與發展線 (Evolutionary Path)
   - 梳理這些論文在時間或技術路徑上的演進關係（例如 A 奠定了基礎，B 解決了 A 的運作成本問題，C 則延伸到跨模態）。
   - 說明技術改進的關鍵軌跡。

## 4. ⚠️ 共同瓶頸與未來研究缺口 (Common Gaps & Limitations)
   - 總結現有這些方法面臨的共同局限性（例如對長文本處理效率低、需要大量標註數據、訓練資源昂貴等）。
   - 點出尚未被完全解決的問題（即研究缺口，Research Gap）。

## 5. 💡 總結與學習建議 (Conclusion & Academic Advice)
   - 為大二資工系學生給出具體的學習與研究方向建議。

## 6. 📖 參考文獻 (References)
   - 以標準學術格式列出這些文獻，例如：
     - `[1] 作者群. "論文標題", 原始檔名.`
     
---

請開始撰寫這份文獻回顧報告："""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            from src.utils.retry_handler import retry_on_429
            response = retry_on_429(self.llm.invoke, messages)
            logger.info("成功生成文獻回顧綜述報告。")
            return response.content
        except Exception as e:
            logger.error(f"呼叫 Gemini 生成文獻回顧綜述失敗: {e}")
            raise e

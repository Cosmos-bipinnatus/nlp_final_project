# -*- coding: utf-8 -*-
"""
Literature Reviewer 全域設定檔
集中管理所有模型名稱、檔案路徑、RAG 切塊參數以及其他全域配置。
"""

from pathlib import Path

# 1. 基礎路徑配置
# 專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 原始論文 PDF 存放目錄
DATA_DIR = BASE_DIR / "data"

# 生成報告存放目錄
DOCS_DIR = BASE_DIR / "docs"

# ChromaDB 本地向量庫儲存目錄
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# 確保必要目錄存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# 3. LLM 模型配置
# 預設使用的 Chat LLM 模型 (改回使用 gemini-2.5-flash 以在您的 Free Tier 專案中取得可用配額，並搭配平滑冷卻)
MODEL_NAME: str = "gemini-2.5-flash"

# 預設使用的 Embedding 向量模型
# 註：已正式升級為 Google 官方 GA 版的 gemini-embedding-2 (models/gemini-embedding-2)，輸出 3072 維的高品質語意向量
EMBEDDING_MODEL: str = "models/gemini-embedding-2"

# 4. RAG & 文本切塊配置
# 每個文字切塊的最大字數（字元數）
CHUNK_SIZE: int = 600

# 相鄰切塊之間的重疊字數
CHUNK_OVERLAP: int = 120

# 向量庫集合名稱（ChromaDB Collection Name）
COLLECTION_NAME: str = "literature_review"

# 5. 工具與 API 配置
# ArXiv 線上檢索的最大論文筆數
ARXIV_MAX_RESULTS: int = 3

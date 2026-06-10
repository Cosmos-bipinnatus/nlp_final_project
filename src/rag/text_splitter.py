import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import get_logger

# 設定日誌
logger = get_logger(__name__)

class AcademicTextSplitter:
    """
    學術論文文本切塊器。
    基於 LangChain 的 RecursiveCharacterTextSplitter 進行語意切塊，
    保留段落、句子完整性，並完美繼承原始頁碼與檔名的 Metadata。
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        初始化切塊器。
        
        Args:
            chunk_size (int): 每個文字切塊的最大字數（字元數）
            chunk_overlap (int): 相鄰切塊之間的重疊字數，用以保留邊界語意上下文
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化 LangChain 的遞迴字元切塊器
        # 預設分隔符順序為: 雙換行(段落) -> 單換行(句子) -> 空格(單字) -> 空字元(字元)
        # 這能確保文本盡可能在「段落」或「句子」層級被切開，而非斷在句子中間。
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"初始化 AcademicTextSplitter: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_parsed_documents(self, parsed_docs: List[Dict[str, Any]]) -> List[Document]:
        """
        將解析後的原始頁面字典清單，轉換為 LangChain Document 物件並進行切塊。
        
        Args:
            parsed_docs (List[Dict[str, Any]]): pdf_parser 解析出來的字典清單。
                                               每個元素必須包含 "page_content" 與 "metadata"。
            
        Returns:
            List[Document]: 切塊後的 LangChain Document 物件清單，
                            每個 Document 的 metadata 皆會繼承原始頁碼與來源檔名。
        """
        logger.info(f"開始對 {len(parsed_docs)} 個原始頁面進行文本切塊...")
        
        # 1. 將我們的原始字典包裝成 LangChain 的 Document 標準格式
        langchain_docs = []
        for doc_dict in parsed_docs:
            page_content = doc_dict.get("page_content", "")
            metadata = doc_dict.get("metadata", {})
            
            # 防呆：避免處理空白頁面
            if not page_content.strip():
                continue
                
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            langchain_docs.append(doc)
            
        try:
            # 2. 呼叫切塊器進行切分
            # LangChain 會在切塊過程中自動為每一小塊複製對應的 metadata
            chunked_docs = self.splitter.split_documents(langchain_docs)
            logger.info(f"切塊完成！原始頁面數: {len(langchain_docs)} -> 生成切塊數: {len(chunked_docs)}")
            
            # 3. 額外補強：為每個切塊加上一個標示自身在列表中位置的 chunk_index (輔助除錯與溯源)
            for idx, chunk in enumerate(chunked_docs):
                chunk.metadata["chunk_index"] = idx
                
            return chunked_docs
            
        except Exception as e:
            logger.error(f"文本切塊過程中發生異常: {e}")
            raise e

# ==========================================
# 驗證與測試模組
# ==========================================
if __name__ == "__main__":
    # 單獨執行此腳本時，進行切塊模擬測試
    print("=== 開始文本切塊器單元測試 ===")
    
    # 模擬兩頁解析後的論文數據
    mock_parsed_docs = [
        {
            "page_content": (
                "Section 1: Introduction\n\n"
                "Artificial intelligence (AI) has experienced massive growth over the past decade. "
                "Deep learning models, particularly neural networks, have revolutionized natural language "
                "processing (NLP) and computer vision tasks. The transformer architecture, first introduced "
                "in 2017, has become the dominant paradigm for modern generative language models.\n\n"
                "In this paper, we explore the capabilities of retrieval-augmented generation (RAG) systems "
                "applied to academic research. We address key challenges such as context window limits, "
                "double-column layout parsing, and precise source attribution."
            ),
            "metadata": {"source": "mock_paper.pdf", "page": 1, "total_pages": 2}
        },
        {
            "page_content": (
                "Section 2: Methodology\n\n"
                "Our proposed architecture consists of three core components: an ingest pipeline, a vectorstore "
                "retriever, and a generation agent. The ingest pipeline reads PDFs, splits texts, and "
                "stores embeddings in ChromaDB.\n\n"
                "We use Google's gemini-embedding-2-preview model to convert text chunks into dense vectors. "
                "These vectors are then queried at runtime to retrieve relevant context segments."
            ),
            "metadata": {"source": "mock_paper.pdf", "page": 2, "total_pages": 2}
        }
    ]
    
    # 初始化切塊器，使用較小的字數以利觀察切分效果
    splitter = AcademicTextSplitter(chunk_size=150, chunk_overlap=30)
    
    try:
        chunks = splitter.split_parsed_documents(mock_parsed_docs)
        print(f"\n成功切出 {len(chunks)} 個切塊：")
        
        for idx, chunk in enumerate(chunks):
            print(f"\n--- [切塊 {idx}] 來源: {chunk.metadata['source']} | 頁碼: {chunk.metadata['page']} | 索引: {chunk.metadata['chunk_index']} ---")
            print(f"長度: {len(chunk.page_content)} 字元")
            print(repr(chunk.page_content))
            
        print("\n✅ 文本切塊功能驗證成功！")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

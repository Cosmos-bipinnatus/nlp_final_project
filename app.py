import streamlit as st
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 匯入 Week 2, Week 3, Week 4 和 Week 5 的核心 NLP/RAG/Agent 元件
from src.loaders.pdf_parser import DoubleColumnPDFParser
from src.rag.text_splitter import AcademicTextSplitter
from src.rag.vector_manager import AcademicVectorManager
from src.rag.generator import AcademicRAGGenerator
from src.agents.router_agent import AcademicRouterAgent
from src.rag.comparison_manager import AcademicComparisonManager


# ==========================================
# 1. 全域初始化與日誌設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入 .env 檔案中的環境變數
load_dotenv()

# 設定 Streamlit 頁面屬性（必須放在最前面）
st.set_page_config(
    page_title="Literature Reviewer - 學術論文文獻回顧系統",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定義上傳論文的儲存目錄（使用 pathlib 確保跨平台相容）
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. 客製化 CSS 樣式 (極致視覺美感 - Glassmorphism 與流暢動畫)
# ==========================================
st.markdown("""
<style>
    /* 引入 Google Fonts 現代字體 */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Noto Sans TC', sans-serif;
    }
    
    /* 頂部極致漸層橫幅 */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 35px 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #a5f3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 10px;
        color: #94a3b8;
    }
    
    /* 玻璃擬態卡片 (Glassmorphism Card) */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(6, 182, 212, 0.3);
    }
    
    /* 徽章樣式 */
    .badge {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.3);
    }
    
    /* 側邊欄漸層裝飾 */
    .sidebar-title {
        font-weight: 700;
        background: linear-gradient(to right, #38bdf8, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 流暢微動畫與按鈕調校 */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* 向量化按鈕特殊發光效果 */
    .vectorize-btn>div>button {
        background: linear-gradient(135deg, #06b6d4 0%, #0d9488 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2) !important;
    }
    .vectorize-btn>div>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4) !important;
    }
    
    /* 清空按鈕樣式 */
    .clear-btn>div>button {
        background: rgba(239, 68, 68, 0.1) !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
    .clear-btn>div>button:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 初始化 RAG 向量管理員與問答生成器
#    使用 @st.cache_resource 進行單例快取：
#    Streamlit 每次使用者互動都會從頭重跑整份 app.py，
#    若不做快取，每次按按鈕都會重新初始化 LLM、Embeddings、ChromaDB 連線，
#    導致免費方案的 Gemini API 迅速達到速率上限 (20 RPM)。
#    使用 @st.cache_resource 後，這些重型物件在整個 Session 中只會被建立一次！
# ==========================================

@st.cache_resource
def init_vector_manager():
    """快取初始化向量管理員（Embeddings + ChromaDB 連線）"""
    return AcademicVectorManager(persist_directory="vectorstore")

@st.cache_resource
def init_rag_generator():
    """快取初始化 RAG 生成器（Gemini 2.5-Flash LLM）"""
    return AcademicRAGGenerator()

@st.cache_resource
def init_academic_agent():
    """快取初始化學術路由代理（Router Agent + Local/ArXiv 工具）"""
    return AcademicRouterAgent(persist_directory="vectorstore")

@st.cache_resource
def init_comparison_manager():
    """快取初始化跨文獻比較管理員（Pydantic 結構化特徵提取引擎）"""
    return AcademicComparisonManager(persist_directory="vectorstore")

# 使用快取函式進行安全初始化
vector_manager = None
rag_generator = None
academic_agent = None
comparison_manager = None
db_error_msg = ""
rag_error_msg = ""
agent_error_msg = ""
comparison_error_msg = ""

try:
    vector_manager = init_vector_manager()
except Exception as e:
    db_error_msg = str(e)
    logger.error(f"Streamlit 初始化向量資料庫失敗: {e}")

try:
    rag_generator = init_rag_generator()
except Exception as e:
    rag_error_msg = str(e)
    logger.error(f"Streamlit 初始化 RAG 生成器失敗: {e}")

try:
    academic_agent = init_academic_agent()
except Exception as e:
    agent_error_msg = str(e)
    logger.error(f"Streamlit 初始化學術路由代理失敗: {e}")

try:
    comparison_manager = init_comparison_manager()
except Exception as e:
    comparison_error_msg = str(e)
    logger.error(f"Streamlit 初始化比較管理員失敗: {e}")


# ==========================================
# 4. 核心業務邏輯：多 PDF 安全上傳與存檔
# ==========================================
def save_uploaded_files(uploaded_files: list) -> tuple[int, int, list[str]]:
    """
    安全地儲存使用者上傳的多個 PDF 檔案至本機 data/ 資料夾。
    """
    saved_count = 0
    skipped_count = 0
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        try:
            if not uploaded_file.name.lower().endswith(".pdf"):
                st.warning(f"⚠️ 跳過非 PDF 檔案: `{uploaded_file.name}`")
                continue
            
            target_path = DATA_DIR / uploaded_file.name
            
            # 檢查是否已存在同名檔案
            if target_path.exists():
                skipped_count += 1
                saved_paths.append(str(target_path))
                continue
            
            # 二進位安全寫入
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            saved_count += 1
            saved_paths.append(str(target_path))
            logger.info(f"檔案寫入成功: {uploaded_file.name} -> {target_path}")
            
        except Exception as e:
            st.error(f"❌ 儲存檔案 `{uploaded_file.name}` 時發生錯誤: {e}")
            logger.error(f"檔案儲存失敗: {uploaded_file.name}, 錯誤: {e}")
            
    return saved_count, skipped_count, saved_paths

# ==========================================
# 5. Streamlit 主視覺 UI 排版
# ==========================================

# 頂部視覺 Banner
st.markdown("""
<div class="hero-container">
    <span class="badge">Week 4: RAG 管線與引用標記 (RAG Pipeline & Citations)</span>
    <h1 class="hero-title">Literature Reviewer 📚</h1>
    <p class="hero-subtitle">大二資工系專題：基於 Google Gemini 2.5-Flash 與雙欄排版還原的學術文獻 RAG 系統</p>
</div>
""", unsafe_allow_html=True)

# 左右雙欄配置 (側邊欄 Sidebar + 主面板 Main Panel)
with st.sidebar:
    st.markdown("### 🛠️ <span class='sidebar-title'>專案控制台</span>", unsafe_allow_html=True)
    st.info("這裡可上傳文獻、啟動/重置向量引擎，並動態監控系統狀態。")
    
    # --- 1. 金鑰狀態區 ---
    st.markdown("---")
    st.markdown("#### 🔑 金鑰狀態")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        st.error("❌ Gemini API Key 未設定")
        st.caption("請在專案根目錄的 `.env` 檔案中設定有效的 `GEMINI_API_KEY`。")
    else:
        st.success("🟢 Gemini API Key 已載入")
        st.caption("成功連結 Google Developer API，提供文字嵌入與生成功能。")
        
    # --- 2. 向量引擎狀態區 ---
    st.markdown("---")
    st.markdown("#### 🧠 向量庫引擎狀態")
    if vector_manager:
        try:
            chunk_count = vector_manager.get_collection_count()
            if chunk_count > 0:
                st.success(f"🟢 已啟用 | 累計 {chunk_count} 個切塊")
            else:
                st.warning("⚪ 已連結 | 目前資料庫為空")
        except Exception as e:
            st.error(f"❌ 資料庫異常: {e}")
    else:
        st.error("❌ 向量庫未啟用")
        st.caption(f"錯誤原因：{db_error_msg}")

    # --- 3. RAG 生成引擎狀態區 ---
    st.markdown("---")
    st.markdown("#### 🤖 AI 生成問答狀態")
    if rag_generator:
        st.success(f"🟢 已啟用 | 模型: {rag_generator.model_name}")
    else:
        st.error("❌ 生成引擎未啟用")
        st.caption(f"錯誤原因：{rag_error_msg}")

    # --- 3-b. 學術路由代理狀態區 (Week 5) ---
    st.markdown("---")
    st.markdown("#### 🧭 學術路由代理狀態")
    if academic_agent:
        st.success(f"🟢 已啟用 | 模式: Pydantic 結構化路由")
    else:
        st.error("❌ 路由代理未啟用")
        st.caption(f"錯誤原因：{agent_error_msg}")

    # --- 3-c. 跨文獻比較矩陣狀態區 (Week 6) ---
    st.markdown("---")
    st.markdown("#### 📊 比較矩陣引擎狀態")
    if comparison_manager:
        st.success(f"🟢 已啟用 | 模式: Pydantic 結構化特徵提取")
    else:
        st.error("❌ 比較引擎未啟用")
        st.caption(f"錯誤原因：{comparison_error_msg}")


    # --- 4. 本地文獻統計區 ---
    st.markdown("---")
    st.markdown("#### 📂 本地文獻庫統計")
    existing_pdfs = list(DATA_DIR.glob("*.pdf"))
    st.metric(label="已上傳論文 PDF 數", value=len(existing_pdfs))
    
    if len(existing_pdfs) > 0:
        st.write("📄 文獻清單：")
        for i, pdf in enumerate(existing_pdfs, 1):
            st.caption(f"{i}. {pdf.name}")
            
        # --- 5. 向量引擎控制按鈕 ---
        st.markdown("---")
        st.markdown("#### ⚡ 向量引擎控制")
        
        # 向量化本地庫按鈕
        if vector_manager:
            st.markdown('<div class="vectorize-btn">', unsafe_allow_html=True)
            if st.button("🔄 向量化本地文獻庫", key="btn_vectorize", use_container_width=True):
                # 建立多層 Loading 動畫
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.info("🧹 正在初始化乾淨的向量空間...")
                    vector_manager.clear_database()
                    progress_bar.progress(10)
                    
                    all_chunks = []
                    total_files = len(existing_pdfs)
                    
                    for idx, pdf_path in enumerate(existing_pdfs, 1):
                        status_text.info(f"📖 正在解析雙欄排版 ({idx}/{total_files}): `{pdf_path.name}`...")
                        parser = DoubleColumnPDFParser(pdf_path)
                        parsed_pages = parser.parse_pdf()
                        
                        status_text.info(f"✂️ 正在進行語意切塊 ({idx}/{total_files}): `{pdf_path.name}`...")
                        splitter = AcademicTextSplitter(chunk_size=600, chunk_overlap=120)
                        chunks = splitter.split_parsed_documents(parsed_pages)
                        all_chunks.extend(chunks)
                        
                        progress_bar.progress(int(10 + (idx / total_files) * 50))
                        
                    if all_chunks:
                        status_text.info(f"🚀 正在將 {len(all_chunks)} 個切塊批次向量化並寫入 ChromaDB...")
                        # 呼叫 models/gemini-embedding-001 批次向量化
                        vector_manager.store_documents(all_chunks)
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        st.success(f"🎉 成功向量化 {total_files} 篇文獻，生成 {len(all_chunks)} 個語意切塊！")
                        st.balloons()
                        st.rerun()
                    else:
                        st.warning("⚠️ 未能產生有效的文本切塊！")
                        
                except Exception as e:
                    status_text.empty()
                    st.error(f"❌ 向量化失敗: {e}")
                    logger.error(f"向量化失敗: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 清空資料庫按鈕
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            if st.button("🗑️ 清空向量資料庫", key="btn_clear", use_container_width=True):
                try:
                    vector_manager.clear_database()
                    st.success("🧹 向量庫已清空！")
                    st.rerun()
                except Exception as e:
                    st.error(f"清空失敗: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

# 主面板 layout
col_main, col_spacer = st.columns([12, 1])

with col_main:
    # 創建標籤頁：1. 上傳論文 2. 語意相似度檢索 3. AI文獻問答與引用定位 4. 跨文獻比較矩陣
    tab_upload, tab_search, tab_qa, tab_comparison = st.tabs([
        "📥 論文上傳與存檔", 
        "🔍 語意檢索與召回測試 (Semantic Search)", 
        "💬 AI 文獻問答與引用定位 (Literature QA)",
        "📊 跨文獻比較矩陣 (Comparison Grid)"
    ])
    
    # ==========================================
    # Tab 1: 論文上傳與存檔
    # ==========================================
    with tab_upload:
        st.markdown("### 📥 多文獻 PDF 上傳區")
        st.write("請將您想要分析的學術論文（支援雙欄 PDF）拖曳至下方：")
        
        uploaded_files = st.file_uploader(
            label="選擇 PDF 檔案（可複選）",
            type=["pdf"],
            accept_multiple_files=True,
            help="系統會自動保留 PDF 檔名及頁碼，並利用雙欄排序演算法還原正確閱讀順序。"
        )
        
        if uploaded_files:
            with st.spinner("🚀 正在安全寫入論文檔案至 data/ 目錄..."):
                saved, skipped, paths = save_uploaded_files(uploaded_files)
                
            st.markdown("---")
            st.markdown("#### 📊 上傳結果報告")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="🆕 新增存檔成功", value=saved)
            with c2:
                st.metric(label="🔄 重複跳過檔案", value=skipped)
            with c3:
                st.metric(label="📑 待處理文獻總數", value=len(paths))
                
            if saved > 0:
                st.success(f"🎉 成功存檔 {saved} 篇文獻！檔案已安全存放至本機 `{DATA_DIR}/` 目錄。")
                st.info("💡 貼心提示：檔案已存檔！請前往左側控制台，點擊 **「🔄 向量化本地文獻庫」** 即可將論文轉換成語意向量，進行語意檢索。")
                st.balloons()
            elif skipped > 0:
                st.info("ℹ️ 本次上傳的檔案皆已存在於文獻庫中，已為您自動跳過。")

    # ==========================================
    # Tab 2: 語意相似度檢索 (Week 3 核心整合驗證)
    # ==========================================
    with tab_search:
        st.markdown("### 🔍 高維語意相似度搜尋測試")
        st.write("利用 Google Gemini 的高維度詞嵌入模型與本地持久化的 ChromaDB，進行相似度召回測試。這是 RAG (檢索增強生成) 的關鍵基礎！")
        
        if not vector_manager or vector_manager.get_collection_count() == 0:
            st.warning("⚠️ 語意檢索不可用：向量庫目前沒有資料！請先完成論文上傳，並在側邊欄點擊「🔄 向量化本地文獻庫」。")
        else:
            # 檢索控制項
            st.markdown("#### 💡 輸入您的學術提問")
            st.caption("例如：`What is the core idea of Transformer?`、`Explain Multi-Head Attention.`、`What datasets were used in the evaluation?`")
            
            search_query = st.text_input(
                label="查詢文字",
                placeholder="輸入您的學術問題進行檢索...",
                label_visibility="collapsed"
            )
            
            col_k, col_submit = st.columns([2, 10])
            with col_k:
                k_val = st.slider("🎯 召回切塊數 (k)", min_value=1, max_value=8, value=3)
                
            with col_submit:
                st.write("")  # 垂直對齊對齊
                st.write("")
                submit_search = st.button("🔍 開始語意檢索")
                
            if search_query or submit_search:
                if not search_query.strip():
                    st.warning("⚠️ 請輸入有效的查詢文字！")
                else:
                    with st.spinner("🕵️ 正在計算查詢向量並於 ChromaDB 進行相似度比對 (L2 距離度量)..."):
                        try:
                            # 執行相似度檢索，取得 (Document, L2_distance) 清單
                            results = vector_manager.semantic_search(search_query, k=k_val)
                            
                            if results:
                                st.success(f"🎉 檢索完成！共召回前 {len(results)} 個語意最相似的學術切塊：")
                                st.markdown("---")
                                
                                # 使用卡片展示每個召回的切塊
                                for rank, (doc, score) in enumerate(results, 1):
                                    source_name = doc.metadata.get("source", "未知文獻")
                                    page_num = doc.metadata.get("page", "?")
                                    total_pages = doc.metadata.get("total_pages", "?")
                                    chunk_idx = doc.metadata.get("chunk_index", "?")
                                    
                                    st.markdown(f"""
                                    <div class="glass-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
                                            <span style="font-weight: 700; color: #06b6d4; font-size: 1.15rem;">
                                                🏆 Rank #{rank} | 向量 L2 距離: <code style="color: #a5f3fc; background-color: rgba(6, 182, 212, 0.15); padding: 2px 6px; border-radius: 4px;">{score:.4f}</code>
                                            </span>
                                            <span class="badge" style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); font-size: 0.85rem; margin-bottom: 0px;">
                                                📑 {source_name} | 第 {page_num} 頁 (共 {total_pages} 頁)
                                            </span>
                                        </div>
                                        <div style="font-size: 1rem; color: #e2e8f0; line-height: 1.7; background-color: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 8px; border-left: 5px solid #06b6d4; font-family: monospace; white-space: pre-wrap;">
{doc.page_content.strip()}
                                        </div>
                                        <div style="display: flex; justify-content: flex-end; font-size: 0.8rem; color: #64748b; margin-top: 8px;">
                                            <span>本地切塊索引: #{chunk_idx}</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("🔍 在高維語意空間中未發現匹配的文本。")
                        except Exception as e:
                            st.error(f"❌ 相似度檢索發生錯誤: {e}")
                            logger.error(f"檢索錯誤: {e}")

    # ==========================================
    # Tab 3: AI 文獻問答與學術代理系統 (Week 4 & 5 整合驗證)
    # ==========================================
    with tab_qa:
        st.markdown("### 💬 AI 論文文獻問答與學術代理系統")
        st.write("本系統支援傳統的 **本地 RAG (檢索增強生成)** 問答，更可開啟 **AI 學術路由代理 (Router Agent)**。代理會自主分析您的提問，決定路由至「本地文獻庫」或「外接 ArXiv 線上學術網」，並在畫面上完整渲染其思考歷程與決策原因！")
        
        if not vector_manager or vector_manager.get_collection_count() == 0:
            st.warning("⚠️ 問答系統不可用：向量庫目前沒有資料！請先完成論文上傳，並在側邊欄點擊「🔄 向量化本地文獻庫」。")
        elif not rag_generator:
            st.error("⚠️ 生成引擎未啟用，無法進行問答。請檢查 API Key 是否設定正確。")
        else:
            # 檢索控制項
            st.markdown("#### 💡 輸入您的學術提問")
            st.caption("例如本地提問：`What is Multi-Head Attention?` | 外部提問：`Search for latest research on RAG in 2025.`")
            
            qa_query = st.text_input(
                label="問答查詢文字",
                placeholder="輸入您的學術問題進行文獻問答與學術路由檢索...",
                label_visibility="collapsed"
            )
            
            # 學術代理路由 Toggle 開關
            use_agent = st.toggle(
                label="🤖 啟用 AI 學術路由代理 (Agent Router)",
                value=True,
                help="啟用後，AI 會先分析您的問題語意，自動決定要閱讀本地已上傳的文獻 (RAG)，還是外接 ArXiv API 搜尋全世界最新的相關論文！"
            )
            
            col_qa_k, col_qa_submit = st.columns([2, 10])
            with col_qa_k:
                qa_k_val = st.slider("🎯 檢索參考切塊數 (k)", min_value=1, max_value=8, value=4, key="qa_k_slider")
                
            with col_qa_submit:
                st.write("")  # 垂直對齊對齊
                st.write("")
                submit_qa = st.button("💬 啟動 AI 智慧問答", key="qa_submit_btn")
                
            if qa_query or submit_qa:
                if not qa_query.strip():
                    st.warning("⚠️ 請輸入有效的問題內容！")
                else:
                    # 判斷是否使用學術路由代理
                    if use_agent and academic_agent:
                        with st.spinner("🕵️ AI 學術代理正在進行問題分析與路由分發決策..."):
                            try:
                                # 執行端到端的代理決策與工具分發
                                response = academic_agent.route_and_execute(qa_query)
                                
                                # A. 渲染精美的 Glassmorphic 思考歷程面板 (Thinking Process)
                                route_zh = "本地文獻庫 (RAG)" if response["route"] == "local" else "外接 ArXiv 學術網 (API)"
                                st.markdown(f"""
                                <div class="glass-card" style="border: 1px solid rgba(6, 182, 212, 0.4); background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.7) 100%); margin-bottom: 25px; padding: 22px; border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);">
                                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                                        <span style="font-size: 1.5rem;">🤖</span>
                                        <span style="font-size: 1.25rem; font-weight: 700; background: linear-gradient(to right, #38bdf8, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI 學術代理路由決策歷程 (Thinking Process)</span>
                                    </div>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 18px;">
                                        <div style="background: rgba(0, 0, 0, 0.25); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px; font-weight: 600;">🎯 路由分發通道</div>
                                            <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8;">{route_zh}</div>
                                        </div>
                                        <div style="background: rgba(0, 0, 0, 0.25); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px; font-weight: 600;">🔑 提取並優化之學術關鍵字</div>
                                            <div style="font-size: 1.05rem; font-weight: 700; color: #a5f3fc; font-family: monospace;">'{response["search_query"]}'</div>
                                        </div>
                                    </div>
                                    <div style="background: rgba(0, 0, 0, 0.2); padding: 16px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); border-left: 4px solid #06b6d4;">
                                        <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 6px; font-weight: 600;">💡 代理決策動機 (Rationale)</div>
                                        <div style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.6;">{response["rationale"]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # B. 渲染最終結果與對應的細節展開區
                                st.success("🎉 AI 學術分析與檢索完成！")
                                st.markdown("#### 📚 學術文獻回顧生成報告")
                                
                                st.markdown(f"""
                                <div class="glass-card" style="border-left: 6px solid #06b6d4; background-color: rgba(15, 23, 42, 0.45); padding: 25px; margin-bottom: 20px;">
                                    <div style="font-size: 1.05rem; line-height: 1.8; color: #f8fafc;">
                                        {response["answer"]}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # C. 根據路由展示各自的參考文獻細節面板
                                if response["route"] == "local" and response["chunks"]:
                                    with st.expander("🔍 檢視本次回答參考之本地原始文獻來源與切塊內容", expanded=False):
                                        for idx, (doc, score) in enumerate(response["chunks"], 1):
                                            source_name = doc.metadata.get("source", "未知文獻")
                                            page_num = doc.metadata.get("page", "?")
                                            total_pages = doc.metadata.get("total_pages", "?")
                                            chunk_idx = doc.metadata.get("chunk_index", "?")
                                            
                                            st.markdown(f"""
                                            <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);">
                                                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #38bdf8; margin-bottom: 5px;">
                                                    <span><b>來源 #{idx}：</b><code>{source_name}</code> (第 {page_num} 頁 / 共 {total_pages} 頁)</span>
                                                    <span>L2 距離：<code>{score:.4f}</code> | 切塊索引：#{chunk_idx}</span>
                                                </div>
                                                <div style="font-size: 0.9rem; color: #94a3b8; font-family: monospace; white-space: pre-wrap; background-color: rgba(0,0,0,0.15); padding: 8px; border-radius: 4px;">
{doc.page_content.strip()}
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                elif response["route"] == "arxiv" and response["papers"]:
                                    with st.expander(f"🌐 檢視 ArXiv 線上論文原始結構化資料（共 {len(response['papers'])} 篇）", expanded=False):
                                        for idx, paper in enumerate(response["papers"], 1):
                                            st.markdown(f"""
                                            <div class="glass-card" style="margin-bottom: 12px; padding: 18px; background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.05);">
                                                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; margin-bottom: 8px; gap: 10px;">
                                                    <span style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">📄 {paper['title']}</span>
                                                    <span class="badge" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); margin-bottom: 0px; font-size: 0.75rem;">ID: {paper['arxiv_id']}</span>
                                                </div>
                                                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">
                                                    👥 <b>作者：</b>{paper['authors']} | 📅 <b>發表日期：</b>{paper['published']}
                                                </div>
                                                <div style="font-size: 0.9rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.45); padding: 12px; border-radius: 6px; font-family: monospace; max-height: 180px; overflow-y: auto; margin-bottom: 10px; line-height: 1.5; border-left: 3px solid #10b981;">
                                                    <b>英文原始摘要：</b><br>{paper['summary']}
                                                </div>
                                                <div style="display: flex; justify-content: flex-end;">
                                                    <a href="{paper['pdf_url']}" target="_blank" style="text-decoration: none; background: rgba(6, 182, 212, 0.12); color: #06b6d4; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; border: 1px solid rgba(6, 182, 212, 0.25); font-weight: 600; transition: all 0.3s;">🔗 前往 ArXiv 下載與閱讀 PDF 檔案</a>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                            except Exception as e:
                                st.error(f"❌ 智慧代理執行失敗: {e}")
                                logger.error(f"智慧代理執行失敗: {e}")
                                
                    else:
                        # 走傳統的單純本地 RAG 模式
                        with st.spinner("🔍 正在高維向量空間中檢索本地最相關之文獻段落..."):
                            try:
                                retrieved_chunks = vector_manager.semantic_search(qa_query, k=qa_k_val)
                            except Exception as e:
                                st.error(f"❌ 檢索失敗: {e}")
                                retrieved_chunks = []
                                
                        if retrieved_chunks:
                            with st.spinner("🤖 正在調用 Gemini 2.5-Flash 進行學術語意分析與引用追蹤..."):
                                try:
                                    cited_answer = rag_generator.generate_answer(qa_query, retrieved_chunks)
                                    
                                    st.success("🎉 本地 RAG 學術分析完成！")
                                    st.markdown("#### 📚 本地 RAG 學術回顧分析結果")
                                    
                                    st.markdown(f"""
                                    <div class="glass-card" style="border-left: 6px solid #06b6d4; background-color: rgba(15, 23, 42, 0.45); padding: 25px;">
                                        <div style="font-size: 1.05rem; line-height: 1.8; color: #f8fafc;">
                                            {cited_answer}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    with st.expander("🔍 檢視本次回答參考之本地原始文獻來源與切塊內容", expanded=False):
                                        for idx, (doc, score) in enumerate(retrieved_chunks, 1):
                                            source_name = doc.metadata.get("source", "未知文獻")
                                            page_num = doc.metadata.get("page", "?")
                                            total_pages = doc.metadata.get("total_pages", "?")
                                            chunk_idx = doc.metadata.get("chunk_index", "?")
                                            
                                            st.markdown(f"""
                                            <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);">
                                                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #38bdf8; margin-bottom: 5px;">
                                                    <span><b>來源 #{idx}：</b><code>{source_name}</code> (第 {page_num} 頁 / 共 {total_pages} 頁)</span>
                                                    <span>L2 距離：<code>{score:.4f}</code> | 切塊索引：#{chunk_idx}</span>
                                                </div>
                                                <div style="font-size: 0.9rem; color: #94a3b8; font-family: monospace; white-space: pre-wrap; background-color: rgba(0,0,0,0.15); padding: 8px; border-radius: 4px;">
{doc.page_content.strip()}
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                except Exception as e:
                                    st.error(f"❌ 生成回答失敗: {e}")
                        else:
                            st.info("🔍 在高維語意空間中未發現匹配的文獻切塊，拒絕生成以免產生幻覺。")

    # ==========================================
    # Tab 4: 跨文獻比較矩陣 (Week 6 核心整合)
    # ==========================================
    with tab_comparison:
        st.markdown("### 📊 跨文獻學術比較矩陣 (Comparison Grid)")
        st.write("本功能支援同時選擇多篇已上傳並向量化的論文。系統會針對每篇論文，透過 **主題式 RAG 檢索** 召回核心章節，並使用 **Pydantic 結構化提煉** 提取出每篇論文的研究方法、實驗資料集與優缺點，最終生成整齊直觀的交叉對照矩陣！")
        
        # 1. 檢查是否有足夠的 PDF 論文
        existing_pdfs = list(DATA_DIR.glob("*.pdf"))
        
        if not vector_manager or vector_manager.get_collection_count() == 0:
            st.warning("⚠️ 比較系統不可用：向量庫目前沒有資料！請先完成論文上傳，並在側邊欄點擊「🔄 向量化本地文獻庫」。")
        elif len(existing_pdfs) < 2:
            st.info("💡 跨文獻交叉比較需要至少 **2 篇** 以上的已向量化文獻。目前檢測到本地文獻庫中只有 1 篇文獻，請先前往第一分頁上傳更多文獻並點擊向量化。")
        elif not comparison_manager:
            st.error(f"❌ 比較引擎未啟用，無法進行分析。錯誤原因：{comparison_error_msg}")
        else:
            # 2. 提供多選選單
            pdf_names = [pdf.name for pdf in existing_pdfs]
            
            st.markdown("#### 📑 選擇您要比較的學術文獻（最少 2 篇）")
            selected_pdfs = st.multiselect(
                label="選擇論文 PDF 檔案",
                options=pdf_names,
                default=pdf_names[:2] if len(pdf_names) >= 2 else pdf_names,
                help="勾選您希望加入交叉比較矩陣的學術論文。"
            )
            
            # 提供一鍵生成按鈕
            submit_comparison = st.button("📊 啟動跨文獻特徵提取與矩陣生成", key="comparison_submit_btn")
            
            # 初始化快取
            if "comparison_data" not in st.session_state:
                st.session_state.comparison_data = None
            if "comparison_pdf_set" not in st.session_state:
                st.session_state.comparison_pdf_set = set()
                
            # 觸發生成
            if submit_comparison:
                if len(selected_pdfs) < 2:
                    st.warning("⚠️ 請至少選擇兩篇文獻以進行交叉比對！")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        comparison_data = []
                        total_pdfs = len(selected_pdfs)
                        
                        for idx, pdf_name in enumerate(selected_pdfs, 1):
                            status_text.info(f"🧬 正在從 `{pdf_name}` 檢索主題切塊並提煉學術特徵 ({idx}/{total_pdfs})...")
                            # 提取單篇特徵
                            features = comparison_manager.extract_features_for_paper(pdf_name)
                            comparison_data.append({
                                "pdf_file": pdf_name,
                                "title": features.title,
                                "methodology": features.methodology,
                                "datasets": features.datasets,
                                "pros": features.pros,
                                "cons": features.cons
                            })
                            progress_bar.progress(int((idx / total_pdfs) * 100))
                            
                        # 儲存至 Session State 快取
                        st.session_state.comparison_data = comparison_data
                        st.session_state.comparison_pdf_set = set(selected_pdfs)
                        
                        status_text.empty()
                        progress_bar.empty()
                        st.success("🎉 跨文獻比較矩陣生成成功！已安全緩存特徵。")
                        st.balloons()
                        
                    except Exception as e:
                        status_text.empty()
                        progress_bar.empty()
                        st.error(f"❌ 比較矩陣生成失敗: {e}")
                        logger.error(f"比較矩陣生成失敗: {e}")
                        
            # 3. 渲染結果
            # 如果 Session State 內有資料，且與當前勾選的文獻集一致（快取擊中）
            if st.session_state.comparison_data and set(selected_pdfs) == st.session_state.comparison_pdf_set:
                data = st.session_state.comparison_data
                
                st.markdown("---")
                st.markdown("#### 📊 跨論文學術比較對照矩陣")
                
                # 將 Dict 轉為 Pandas DataFrame
                import pandas as pd
                df = pd.DataFrame(data)
                
                # 重新命名字段以求美觀
                df_display = df.rename(columns={
                    "pdf_file": "📄 原始 PDF 檔名",
                    "title": "📌 論文正式名稱 (Title)",
                    "methodology": "🛠️ 核心研究方法 (Methodology)",
                    "datasets": "📊 實驗資料集與指標 (Datasets & Metrics)",
                    "pros": "👍 主要創新與優點 (Pros)",
                    "cons": "👎 局限性與缺點 (Cons)"
                })
                
                # 渲染為 Streamlit 表格
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # 4. 提供 Markdown 表格下載功能
                st.markdown("---")
                st.markdown("#### 📥 匯出學術比較報告")
                st.caption("您可以一鍵複製或下載標準學術 Markdown 格式的比較表格，直接貼入您的期末報告中。")
                
                md_table = comparison_manager.convert_to_markdown_table(data)
                
                # 顯示 Markdown
                st.code(md_table, language="markdown")
                
                # 提供下載
                st.download_button(
                    label="📥 下載 Markdown 表格檔案",
                    data=md_table,
                    file_name="literature_comparison_matrix.md",
                    mime="text/markdown",
                    key="download_md_btn"
                )

# 系統底部狀態資訊
st.markdown("---")
st.caption("Literature Reviewer - Version 1.0.0 (Gemini Edition) | Developed by Sophomore CS Students with AI Mentor")

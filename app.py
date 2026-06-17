import streamlit as st
import os
import logging
from pathlib import Path

# 匯入核心 NLP/RAG/Agent 元件
from src.loaders.pdf_parser import DoubleColumnPDFParser
from src.rag.text_splitter import AcademicTextSplitter
from src.rag.vector_manager import AcademicVectorManager
from src.rag.generator import AcademicRAGGenerator
from src.agents.router_agent import AcademicRouterAgent
from src.rag.comparison_manager import AcademicComparisonManager
from src.config import DATA_DIR, VECTORSTORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, EMBEDDING_MODEL
from src.ui import render_styles, render_hero_banner


# ==========================================
# 1. 全域初始化與日誌設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 設定 Streamlit 頁面屬性（預設摺疊側邊欄以最大化主介面空間）
st.set_page_config(
    page_title="Literature Reviewer - 學術論文文獻回顧系統",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 確保必要目錄存在 (已在 config.settings 初始化)

# ==========================================
# 2. 客製化 CSS 樣式 (極致視覺美感 - Glassmorphism 與流暢動畫)
# ==========================================
render_styles()

# 初始化全域 Session State 變數
if "sidebar_active" not in st.session_state:
    st.session_state["sidebar_active"] = False

if "gemini_api_key_val" not in st.session_state:
    st.session_state["gemini_api_key_val"] = ""

if "show_api_key_input" not in st.session_state:
    st.session_state["show_api_key_input"] = True

# ==========================================
# 3. 初始化 RAG 向量管理員與問答生成器
#    使用 @st.cache_resource 進行單例快取：
#    Streamlit 每次使用者互動都會從頭重跑整份 app.py，
#    若不做快取，每次按按鈕都會重新初始化 LLM、Embeddings、ChromaDB 連線，
#    導致免費方案的 Gemini API 迅速達到速率上限 (20 RPM)。
#    使用 @st.cache_resource 後，這些重型物件在整個 Session 中只會被建立一次！
# ==========================================

@st.cache_resource
def init_vector_manager(api_key: str, embedding_model: str):
    """快取初始化向量管理員（Embeddings + ChromaDB 連線）"""
    os.environ["GEMINI_API_KEY"] = api_key
    return AcademicVectorManager(persist_directory=VECTORSTORE_DIR)

@st.cache_resource
def init_rag_generator(api_key: str, model_name: str):
    """快取初始化 RAG 生成器（Gemini 2.5-Flash LLM）"""
    os.environ["GEMINI_API_KEY"] = api_key
    return AcademicRAGGenerator(model_name=model_name)

@st.cache_resource
def init_academic_agent(api_key: str, model_name: str):
    """快取初始化學術路由代理（Router Agent + Local/ArXiv 工具）"""
    os.environ["GEMINI_API_KEY"] = api_key
    return AcademicRouterAgent(model_name=model_name, persist_directory=str(VECTORSTORE_DIR))

@st.cache_resource
def init_comparison_manager(api_key: str, model_name: str, _vector_manager=None):
    """快取初始化跨文獻比較管理員（Pydantic 結構化特徵提取引擎）"""
    os.environ["GEMINI_API_KEY"] = api_key
    return AcademicComparisonManager(persist_directory=str(VECTORSTORE_DIR), vector_manager=_vector_manager)

# 從 Session State 取得使用者輸入的 API 金鑰以啟用後續引擎初始化
gemini_api_key = st.session_state.get("gemini_api_key_val", "")

# 使用快取函式進行安全初始化
vector_manager = None
rag_generator = None
academic_agent = None
comparison_manager = None
db_error_msg = ""
rag_error_msg = ""
agent_error_msg = ""
comparison_error_msg = ""

if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    try:
        vector_manager = init_vector_manager(gemini_api_key, EMBEDDING_MODEL)
    except Exception as e:
        db_error_msg = str(e)
        logger.error(f"Streamlit 初始化向量資料庫失敗: {e}")

    try:
        rag_generator = init_rag_generator(gemini_api_key, MODEL_NAME)
    except Exception as e:
        rag_error_msg = str(e)
        logger.error(f"Streamlit 初始化 RAG 生成器失敗: {e}")

    try:
        academic_agent = init_academic_agent(gemini_api_key, MODEL_NAME)
    except Exception as e:
        agent_error_msg = str(e)
        logger.error(f"Streamlit 初始化學術路由代理失敗: {e}")

    try:
        comparison_manager = init_comparison_manager(gemini_api_key, MODEL_NAME, _vector_manager=vector_manager)
    except Exception as e:
        comparison_error_msg = str(e)
        logger.error(f"Streamlit 初始化比較管理員失敗: {e}")



# --- 3-d. 骨架屏載入動畫渲染器 (Week 6 視覺優化) ---
def render_skeleton_screen(card_type="qa"):
    """渲染具有脈動動畫的 HTML 骨架屏"""
    if card_type == "qa":
        return """
        <div class="skeleton-card">
            <div class="skeleton-title"></div>
            <div class="skeleton-line" style="width: 90%;"></div>
            <div class="skeleton-line" style="width: 85%;"></div>
            <div class="skeleton-line" style="width: 95%;"></div>
            <div class="skeleton-line" style="width: 70%;"></div>
        </div>
        """
    elif card_type == "search":
        return """
        <div style="display: grid; grid-template-columns: 1fr; gap: 15px; width: 100%;">
            <div class="skeleton-card" style="padding: 15px; margin-bottom: 0px;">
                <div class="skeleton-title" style="width: 30%; height: 16px;"></div>
                <div class="skeleton-line" style="width: 95%; height: 12px;"></div>
                <div class="skeleton-line" style="width: 80%; height: 12px;"></div>
            </div>
            <div class="skeleton-card" style="padding: 15px; margin-bottom: 0px;">
                <div class="skeleton-title" style="width: 25%; height: 16px;"></div>
                <div class="skeleton-line" style="width: 90%; height: 12px;"></div>
                <div class="skeleton-line" style="width: 75%; height: 12px;"></div>
            </div>
        </div>
        """
    elif card_type == "comparison":
        return """
        <div class="skeleton-card">
            <div class="skeleton-title" style="width: 40%; height: 20px; margin-bottom: 25px;"></div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
                <div class="skeleton-line" style="height: 100px;"></div>
                <div class="skeleton-line" style="height: 100px;"></div>
                <div class="skeleton-line" style="height: 100px;"></div>
                <div class="skeleton-line" style="height: 100px;"></div>
                <div class="skeleton-line" style="height: 100px;"></div>
            </div>
        </div>
        """
    return ""

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

# 頂部視覺標題 (改用簡潔的無背景漸層文字，避免過度擁擠的卡片)
# 頂部雙欄配置：左側為系統標題與介紹，右側為 API 金鑰輸入與亮綠燈按鈕
col_title, col_key = st.columns([9, 3])

with col_title:
    st.markdown(
        """
        <div style="padding: 10px 0;">
            <h1 style="font-size: 2.6rem; font-weight: 800; margin: 0; background: linear-gradient(to right, #38bdf8, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Literature Reviewer 📚
            </h1>
            <p style="font-size: 1.05rem; color: #94a3b8; margin-top: 6px; margin-bottom: 0;">
                大二資工系專題：基於 Google Gemini 2.5-Flash 與雙欄排版還原的學術文獻 RAG 系統
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_key:
    # 垂直對齊占位
    st.write("")
    st.write("")
    
    # 檢查是否有金鑰值
    has_key = len(st.session_state.get("gemini_api_key_val", "").strip()) > 0
    
    if has_key:
        # 當輸入完成後，顯示亮綠燈 "已啟用" 狀態藥丸按鈕
        st.markdown('<div class="api-status-container">', unsafe_allow_html=True)
        if st.button("🟢 已啟用", key="btn_api_status", use_container_width=True, help="點擊此處展開/隱藏金鑰輸入框以進行修改"):
            st.session_state["show_api_key_input"] = not st.session_state.get("show_api_key_input", False)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 若無金鑰則強制顯示輸入框
        st.session_state["show_api_key_input"] = True

    # 渲染輸入框
    if st.session_state.get("show_api_key_input", True):
        api_key_input = st.text_input(
            label="請輸入您的 Gemini API Key",
            type="password",
            placeholder="請輸入 Gemini API Key 以啟用服務...",
            value=st.session_state.get("gemini_api_key_val", ""),
            key="api_key_input_widget",
            label_visibility="collapsed"
        )
        # 狀態值發生變更則即時更新與重新載入
        if api_key_input != st.session_state.get("gemini_api_key_val", ""):
            st.session_state["gemini_api_key_val"] = api_key_input
            if api_key_input.strip():
                st.session_state["show_api_key_input"] = False
            st.rerun()

# 移除側邊欄的所有監控狀態，使介面極淨化並釋放側邊空間
pass

# 主面板 layout (移除了側欄與右側 spacer 欄位以拉大主面板寬度)
col_main = st.container()

with col_main:
    # 創建標籤頁：1. 上傳論文 2. 語意相似度檢索 3. AI文獻問答與引用定位 4. 跨文獻比較矩陣 5. 文獻回顧綜述
    tab_upload, tab_search, tab_qa, tab_comparison, tab_review = st.tabs([
        "📥 論文上傳與存檔", 
        "🔍 語意檢索與召回測試 (Semantic Search)", 
        "💬 AI 文獻問答與引用定位 (Literature QA)",
        "📊 跨文獻比較矩陣 (Comparison Grid)",
        "📚 文獻回顧綜述 (Literature Review)"
    ])
    
    # ==========================================
    # Tab 1: 論文上傳與存檔
    # ==========================================
    with tab_upload:
        st.markdown("### 📥 多文獻 PDF 上傳與引擎管理區")
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
                st.info("💡 貼心提示：檔案已存檔！請在下方點擊 **「🔄 向量化本地文獻庫」** 即可將論文轉換成語意向量，進行語意檢索。")
                st.balloons()
            elif skipped > 0:
                st.info("ℹ️ 本次上傳的檔案皆已存在於文獻庫中，已為您自動跳過。")

        st.markdown("---")
        
        # 建立全域進度與狀態佔位符，使進度條與警告文字能在外層大容器完整展開，防範窄列（如 15% 寬度）擠壓變形
        global_progress_placeholder = st.empty()
        global_status_placeholder = st.empty()
        
        # 建立左右雙欄：左側為本地文獻統計，右側為向量引擎控制
        col_list, col_engine = st.columns([6, 6])
        
        with col_list:
            st.markdown("#### 📂 本地文獻庫統計")
            existing_pdfs = list(DATA_DIR.glob("*.pdf"))
            st.metric(label="已上傳論文 PDF 數", value=len(existing_pdfs))
            
            if len(existing_pdfs) > 0:
                st.write("📄 文獻清單：")
                
                # 獲取當前已向量化的來源清單
                vectorized_sources = vector_manager.get_unique_sources() if vector_manager else []
                
                for i, pdf in enumerate(existing_pdfs, 1):
                    is_vectorized = pdf.name in vectorized_sources
                    status_emoji = "🟢" if is_vectorized else "⚪"
                    
                    col_pdf_name, col_actions = st.columns([7, 3])
                    
                    with col_pdf_name:
                        st.caption(f"{status_emoji} {i}. {pdf.name}")
                        
                    with col_actions:
                        col_v, col_d = st.columns(2)
                        
                        # A. 向量化按鈕：若尚未向量化，顯示單篇向量化按鈕
                        with col_v:
                            if not is_vectorized:
                                if st.button("🔄", key=f"btn_vec_single_{pdf.name}", help=f"單篇向量化: {pdf.name}"):
                                    progress_bar = global_progress_placeholder.progress(0)
                                    status_text = global_status_placeholder.empty()
                                    try:
                                        status_text.info(f"📖 正在解析雙欄排版...")
                                        parser = DoubleColumnPDFParser(pdf)
                                        parsed_pages = parser.parse_pdf()
                                        progress_bar.progress(30)
                                        
                                        status_text.info(f"✂️ 正在進行語意切塊...")
                                        splitter = AcademicTextSplitter(chunk_size=600, chunk_overlap=120)
                                        chunks = splitter.split_parsed_documents(parsed_pages)
                                        progress_bar.progress(60)
                                        
                                        if chunks:
                                            def single_callback(batch_num, total_batches, remaining_sec, status):
                                                if status == "writing":
                                                    status_text.info(f"🚀 正在將第 **{batch_num}/{total_batches}** 批寫入向量庫，請稍候...")
                                                    progress_bar.progress(int(60 + (batch_num - 1) / total_batches * 38))
                                                elif status == "cooling":
                                                    status_text.warning(
                                                        f"⏳ **流量平滑防護中 (30K TPM 限流避坑)**...\n"
                                                        f"第 **{batch_num}/{total_batches}** 批寫入成功。安全冷卻倒數：**{remaining_sec}** 秒..."
                                                    )
                                                    progress_bar.progress(int(60 + batch_num / total_batches * 38))
                                            vector_manager.store_documents(chunks, progress_callback=single_callback)
                                            progress_bar.progress(100)
                                            st.success(f"🎉 論文 `{pdf.name}` 向量化成功！")
                                            st.balloons()
                                            st.rerun()
                                        else:
                                            st.warning("⚠️ 未能產生有效的文本切塊！")
                                    except Exception as e:
                                        st.error(f"❌ 向量化失敗: {e}")
                                    finally:
                                        global_status_placeholder.empty()
                                        global_progress_placeholder.empty()
                                        progress_bar.empty()
                            else:
                                st.write("") # 佔位
                                
                        # B. 刪除按鈕：物理刪除實體檔案與 ChromaDB 向量，並清理比較矩陣快取
                        with col_d:
                            if st.button("🗑️", key=f"btn_del_single_{pdf.name}", help=f"刪除此論文與向量: {pdf.name}"):
                                try:
                                    # 1. 刪除 ChromaDB 中的向量
                                    if vector_manager:
                                        vector_manager.delete_by_source(pdf.name)
                                    # 2. 物理刪除檔案
                                    if pdf.exists():
                                        pdf.unlink()
                                    # 3. 清理比較對應之 Session State 快取
                                    if "comparison_data" in st.session_state and st.session_state.comparison_data:
                                        st.session_state.comparison_data = [
                                            item for item in st.session_state.comparison_data if item["pdf_file"] != pdf.name
                                        ]
                                        if pdf.name in st.session_state.comparison_pdf_set:
                                            st.session_state.comparison_pdf_set.discard(pdf.name)
                                            
                                    st.success(f"🧹 已成功刪除 `{pdf.name}` 及其向量資料！")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 刪除失敗: {e}")
            else:
                st.info("📂 目前文獻庫中無 PDF 檔案，請在上方上傳。")
                    
        with col_engine:
            st.markdown("#### ⚡ 向量引擎控制")
            
            # 檢查是否有巨型 PDF 論文/書籍 (單篇大於 50 頁)
            has_large_pdf = False
            large_pdf_names = []
            for pdf_path in existing_pdfs:
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    if len(doc) > 50:
                        has_large_pdf = True
                        large_pdf_names.append(pdf_path.name)
                    doc.close()
                except Exception:
                    pass
                    
            if has_large_pdf:
                st.warning(
                    f"💡 **偵測到巨型書籍/文獻**：\n"
                    f"文獻 {', '.join([f'`{n}`' for n in large_pdf_names])} 超過 50 頁。一鍵向量化整本巨著極易擊穿免費 API 配額。\n\n"
                    f"**強烈建議**：使用左側文獻清單個別論文旁的 **「🔄」** 按鈕進行「單篇向量化」，以維持系統穩定。"
                )
            
            # 向量化本地庫按鈕
            if vector_manager:
                st.markdown('<div class="vectorize-btn">', unsafe_allow_html=True)
                if st.button("🔄 向量化本地文獻庫", key="btn_vectorize", use_container_width=True):
                    if not existing_pdfs:
                        st.warning("⚠️ 目前本地文獻目錄 `data/` 中沒有任何 PDF 檔案！請先在左側上傳論文。")
                    else:
                        progress_bar = global_progress_placeholder.progress(0)
                        status_text = global_status_placeholder.empty()
                        
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
                                def all_callback(batch_num, total_batches, remaining_sec, status):
                                    if status == "writing":
                                        status_text.info(f"🚀 正在將第 **{batch_num}/{total_batches}** 批寫入向量庫，共 {len(all_chunks)} 個切塊...")
                                        progress_bar.progress(int(60 + (batch_num - 1) / total_batches * 38))
                                    elif status == "cooling":
                                        status_text.warning(
                                            f"⏳ **流量平滑防護中 (30K TPM 限流避坑)**...\n"
                                            f"第 **{batch_num}/{total_batches}** 批寫入成功。安全冷卻倒數：**{remaining_sec}** 秒..."
                                        )
                                        progress_bar.progress(int(60 + batch_num / total_batches * 38))
                                vector_manager.store_documents(all_chunks, progress_callback=all_callback)
                                
                                progress_bar.progress(100)
                                global_status_placeholder.empty()
                                global_progress_placeholder.empty()
                                st.success(f"🎉 成功向量化 {total_files} 篇文獻，生成 {len(all_chunks)} 個語意切塊！")
                                st.balloons()
                                st.rerun()
                            else:
                                st.warning("⚠️ 未能產生有效的文本切塊！")
                                
                        except Exception as e:
                            global_status_placeholder.empty()
                            global_progress_placeholder.empty()
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
            else:
                st.error("❌ 向量庫未啟用，請在左側輸入有效的 API Key 以啟用服務。")

    # ==========================================
    # Tab 2: 語意相似度檢索 (Week 3 核心整合驗證)
    # ==========================================
    with tab_search:
        st.markdown("### 🔍 高維語意相似度搜尋測試")
        st.write("利用 Google Gemini 的高維度詞嵌入模型與本地持久化的 ChromaDB，進行相似度召回測試。這是 RAG (檢索增強生成) 的關鍵基礎！")
        
        if not gemini_api_key:
            st.info("👉 請在左側「專案控制台」輸入您的 **Gemini API Key** 以啟用語意相似度檢索功能。")
        elif not vector_manager or vector_manager.get_collection_count() == 0:
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
                    placeholder = st.empty()
                    placeholder.markdown(render_skeleton_screen("search"), unsafe_allow_html=True)
                    try:
                        # 執行相似度檢索，取得 (Document, L2_distance) 清單
                        results = vector_manager.semantic_search(search_query, k=k_val)
                        placeholder.empty()
                        
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
                                        <span class="badge" style="background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%); font-size: 0.85rem; margin-bottom: 0px;">
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
    # Tab 3: AI 文獻問答與學術代理系統 (Week 4 & 5 整合，Option A & C 優化)
    # ==========================================
    with tab_qa:
        # 1. 標題與對話清空按鈕雙欄排版
        col_qa_title, col_qa_clear = st.columns([8, 2])
        with col_qa_title:
            st.markdown("### 💬 AI 論文文獻問答與學術代理系統")
        with col_qa_clear:
            if st.button("🗑️ 清空對話紀錄", key="btn_clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
                
        st.write("本系統支援傳統的 **本地 RAG (檢索增強生成)** 問答，更可開啟 **AI 學術路由代理 (Router Agent)**。代理會自主分析您的提問，決定路由至「本地文獻庫」或「外接 ArXiv 線上學術網」，並在畫面上完整渲染其思考歷程與決策原因！")
        
        # 初始化對話快取
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        if not gemini_api_key:
            st.info("👉 請在左側「專案控制台」輸入您的 **Gemini API Key** 以啟用 AI 論文問答功能。")
        elif not vector_manager or vector_manager.get_collection_count() == 0:
            st.warning("⚠️ 問答系統不可用：向量庫目前沒有資料！請先完成論文上傳，並在側邊欄點擊「🔄 向量化本地文獻庫」。")
        elif not rag_generator:
            st.error("⚠️ 生成引擎未啟用，無法進行問答。請檢查 API Key 是否設定正確。")
        else:
            # 2. 控制開關排版
            use_agent = st.toggle(
                label="🤖 啟用 AI 學術路由代理 (Agent Router)",
                value=True,
                help="啟用後，AI 會先分析您的問題語意，自動決定要閱讀本地已上傳的文獻 (RAG)，還是外接 ArXiv API 搜尋全世界最新的相關論文！"
            )
            
            col_qa_k, _ = st.columns([3, 7])
            with col_qa_k:
                qa_k_val = st.slider("🎯 檢索參考切塊數 (k)", min_value=1, max_value=8, value=4, key="qa_k_slider")
            st.markdown("---")
            
            # 3. 渲染對話歷史記錄 (Option A Bubble UI)
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    # 展開思考歷程或參考來源
                    if msg["role"] == "assistant":
                        if "thinking" in msg and msg["thinking"]:
                            with st.expander("🤖 檢視路由決策歷程 (Thinking Process)", expanded=False):
                                st.markdown(msg["thinking"], unsafe_allow_html=True)
                        if "sources" in msg and msg["sources"]:
                            with st.expander("🔍 檢視本次回答參考之學術文獻來源與切塊", expanded=False):
                                for s_html in msg["sources"]:
                                    st.markdown(s_html, unsafe_allow_html=True)
                                    
            # 4. 輸入欄位與生成邏輯 (st.chat_input)
            if qa_query := st.chat_input("輸入您的學術問題進行問答與檢索...", key="qa_chat_input"):
                # 立即在畫面上渲染使用者的提問
                with st.chat_message("user"):
                    st.markdown(qa_query)
                st.session_state.chat_history.append({"role": "user", "content": qa_query})
                
                # 渲染 assistant 回覆氣泡與骨架屏 (Option E)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown(render_skeleton_screen("qa"), unsafe_allow_html=True)
                    
                    try:
                        thinking_html = ""
                        sources_html_list = []
                        answer_content = ""
                        plain_answer = ""
                        
                        if use_agent and academic_agent:
                            # 執行代理路由與工具分發
                            response = academic_agent.route_and_execute(qa_query, chat_history=st.session_state.chat_history[:-1])
                            
                            # 整理 Thinking Process HTML
                            route_map = {
                                "local": "本地文獻庫 (RAG)",
                                "arxiv": "外接 ArXiv 學術網 (API)",
                                "hybrid": "混合對照綜述 (Hybrid RAG)",
                                "direct": "通用學術知識直答 (Direct Response)"
                            }
                            route_zh = route_map.get(response["route"], response["route"])
                            thinking_html = f"""
                            <div style="background: rgba(0, 0, 0, 0.25); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 10px;">
                                <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px; font-weight: 600;">🎯 路由分發通道：<b>{route_zh}</b></div>
                                <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px; font-weight: 600;">🔑 提取並優化之學術關鍵字：<code style="color: #a5f3fc; font-family: monospace;">'{response["search_query"]}'</code></div>
                                <div style="font-size: 0.9rem; color: #cbd5e1; border-left: 3px solid #06b6d4; padding-left: 10px; margin-top: 8px; font-style: italic; line-height: 1.5;">
                                    <b>決策理由：</b>{response["rationale"]}
                                </div>
                            </div>
                            """
                            
                            # 整理 Chunks / Papers HTML
                            if response["route"] == "local" and response["chunks"]:
                                for idx, (doc, score) in enumerate(response["chunks"], 1):
                                    source_name = doc.metadata.get("source", "未知文獻")
                                    page_num = doc.metadata.get("page", "?")
                                    total_pages = doc.metadata.get("total_pages", "?")
                                    chunk_idx = doc.metadata.get("chunk_index", "?")
                                    
                                    block_html = f"""
                                    <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);">
                                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #38bdf8; margin-bottom: 5px;">
                                            <span><b>來源 #{idx}：</b><code>{source_name}</code> (第 {page_num} 頁 / 共 {total_pages} 頁)</span>
                                            <span>L2 距離：<code>{score:.4f}</code> | 切塊索引：#{chunk_idx}</span>
                                        </div>
                                        <div style="font-size: 0.9rem; color: #94a3b8; font-family: monospace; white-space: pre-wrap; background-color: rgba(0,0,0,0.15); padding: 8px; border-radius: 4px;">
                                            {doc.page_content.strip()}
                                        </div>
                                    </div>
                                    """
                                    sources_html_list.append(block_html)
                                    
                            elif response["route"] == "arxiv" and response["papers"]:
                                for idx, paper in enumerate(response["papers"], 1):
                                    block_html = f"""
                                    <div style="margin-bottom: 12px; padding: 14px; background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px;">
                                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; margin-bottom: 8px; gap: 10px;">
                                            <span style="font-weight: 700; color: #38bdf8; font-size: 0.95rem;">📄 {paper['title']}</span>
                                            <span class="badge" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); margin-bottom: 0px; font-size: 0.70rem; padding: 2px 6px;">ID: {paper['arxiv_id']}</span>
                                        </div>
                                        <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;">
                                            👥 <b>作者：</b>{paper['authors']} | 📅 <b>發表日期：</b>{paper['published']}
                                        </div>
                                        <div style="font-size: 0.85rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.45); padding: 10px; border-radius: 6px; font-family: monospace; line-height: 1.4; border-left: 3px solid #10b981;">
                                            <b>英文摘要：</b>{paper['summary']}
                                        </div>
                                        <div style="display: flex; justify-content: flex-end; margin-top: 8px;">
                                            <a href="{paper['pdf_url']}" target="_blank" style="text-decoration: none; background: rgba(6, 182, 212, 0.12); color: #06b6d4; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; border: 1px solid rgba(6, 182, 212, 0.25); font-weight: 600;">🔗 下載 PDF</a>
                                        </div>
                                    </div>
                                    """
                                    sources_html_list.append(block_html)
                                    
                            answer_content = response["answer"]
                            plain_answer = response["answer"]
                            
                        else:
                            # 傳統本地 RAG 模式
                            retrieved_chunks = vector_manager.semantic_search(qa_query, k=qa_k_val)
                            
                            if retrieved_chunks:
                                cited_answer = rag_generator.generate_answer(qa_query, retrieved_chunks, chat_history=st.session_state.chat_history[:-1])
                                answer_content = cited_answer
                                plain_answer = cited_answer
                                
                                for idx, (doc, score) in enumerate(retrieved_chunks, 1):
                                    source_name = doc.metadata.get("source", "未知文獻")
                                    page_num = doc.metadata.get("page", "?")
                                    total_pages = doc.metadata.get("total_pages", "?")
                                    chunk_idx = doc.metadata.get("chunk_index", "?")
                                    
                                    block_html = f"""
                                    <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);">
                                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #38bdf8; margin-bottom: 5px;">
                                            <span><b>來源 #{idx}：</b><code>{source_name}</code> (第 {page_num} 頁 / 共 {total_pages} 頁)</span>
                                            <span>L2 距離：<code>{score:.4f}</code> | 切塊索引：#{chunk_idx}</span>
                                        </div>
                                        <div style="font-size: 0.9rem; color: #94a3b8; font-family: monospace; white-space: pre-wrap; background-color: rgba(0,0,0,0.15); padding: 8px; border-radius: 4px;">
                                            {doc.page_content.strip()}
                                        </div>
                                    </div>
                                    """
                                    sources_html_list.append(block_html)
                            else:
                                answer_content = (
                                    "抱歉，根據目前已向量化的文獻庫，未發現與您的問題直接相關的研究數據或學術結論。\n"
                                    "💡 **建議**：請先前往「上傳文獻區」上傳包含該主題的 PDF 論文，並點擊「向量化本地文獻庫」以利系統檢索。"
                                )
                                plain_answer = answer_content
                                
                        formatted_answer_html = f"""
                        <div class="glass-card" style="border-left: 6px solid #06b6d4; background-color: rgba(15, 23, 42, 0.45); padding: 20px; margin-bottom: 10px;">
                            <div style="font-size: 1.05rem; line-height: 1.8; color: #f8fafc; white-space: pre-wrap;">
                                {answer_content}
                            </div>
                        </div>
                        """
                        
                        placeholder.markdown(formatted_answer_html, unsafe_allow_html=True)
                        
                        if thinking_html:
                            with st.expander("🤖 檢視路由決策歷程 (Thinking Process)", expanded=False):
                                st.markdown(thinking_html, unsafe_allow_html=True)
                        if sources_html_list:
                            with st.expander("🔍 檢視本次回答參考之學術文獻來源與切塊", expanded=False):
                                for s_html in sources_html_list:
                                    st.markdown(s_html, unsafe_allow_html=True)
                                    
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": formatted_answer_html,
                            "plain_answer": plain_answer,
                            "thinking": thinking_html,
                            "sources": sources_html_list
                        })
                        st.rerun()
                        
                    except Exception as e:
                        placeholder.empty()
                        st.error(f"❌ 生成回答失敗: {e}")
                        logger.error(f"生成回答失敗: {e}")
                        
            # 5. PDF 一鍵下載報告 (Option C 優化 - 預設下載 PDF)
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("#### 📥 匯出完整對話紀錄為 PDF 報告")
                st.caption("您可以一鍵下載包含所有引用標記與參考切塊的標準 A4 PDF 學術報告。")
                
                try:
                    from src.utils.pdf_generator import generate_qa_pdf
                    pdf_bytes = generate_qa_pdf(st.session_state.chat_history)
                    
                    st.download_button(
                        label="📥 下載完整學術問答報告 (.pdf)",
                        data=pdf_bytes,
                        file_name="literature_review_qa_report.pdf",
                        mime="application/pdf",
                        key="download_qa_report_pdf_btn",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.error(f"❌ PDF 生成失敗: {pdf_err}")

    # ==========================================
    # Tab 4: 跨文獻比較矩陣 (Week 6 核心整合)
    # ==========================================
    with tab_comparison:
        st.markdown("### 📊 跨文獻學術比較矩陣 (Comparison Grid)")
        st.write("本功能支援同時選擇多篇已上傳並向量化的論文。系統會針對每篇論文，透過 **主題式 RAG 檢索** 召回核心章節，並使用 **Pydantic 結構化提煉** 提取出每篇論文的研究方法、實驗資料集與優缺點，最終生成整齊直觀的交叉對照矩陣！")
        
        # 1. 檢查是否有足夠的 PDF 論文
        existing_pdfs = list(DATA_DIR.glob("*.pdf"))
        
        if not gemini_api_key:
            st.info("👉 請在左側「專案控制台」輸入您的 **Gemini API Key** 以啟用跨文獻比較功能。")
        elif not vector_manager or vector_manager.get_collection_count() == 0:
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
                    
                    # 顯示比較脈動骨架屏 (Option E)
                    skeleton_placeholder = st.empty()
                    skeleton_placeholder.markdown(render_skeleton_screen("comparison"), unsafe_allow_html=True)
                    
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
                        skeleton_placeholder.empty()
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
                
                # 4. 提供 PDF 報告與 Markdown 表格下載功能 (預設下載 PDF)
                st.markdown("---")
                st.markdown("#### 📥 匯出學術比較報告")
                st.caption("您可以一鍵下載包含所有文獻比較資料的標準 A4 PDF 報告。底下的 Markdown 表格代碼可供您複製貼上。")
                
                try:
                    from src.utils.pdf_generator import generate_comparison_pdf
                    pdf_bytes = generate_comparison_pdf(data)
                    
                    st.download_button(
                        label="📥 下載學術比較報告 (.pdf)",
                        data=pdf_bytes,
                        file_name="literature_comparison_report.pdf",
                        mime="application/pdf",
                        key="download_pdf_btn",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.error(f"❌ PDF 生成失敗: {pdf_err}")
                
                # 同時依然保留 Markdown 表格預覽，方便學生複製
                st.markdown("##### 📄 複製 Markdown 格式表格")
                md_table = comparison_manager.convert_to_markdown_table(data)
                st.code(md_table, language="markdown")

    # ==========================================
    # Tab 5: 文獻回顧報告生成 (B2 & Week 6 綜述生成)
    # ==========================================
    with tab_review:
        st.markdown("### 📚 多文獻學術綜述與研究回顧報告生成 (Literature Review)")
        st.write("本功能支援一鍵為多篇論文生成整合性的學術綜述與文獻回顧報告。系統將利用特徵提取引擎對選定論文進行交叉分析，提煉出技術演進脈絡、共同瓶頸與研究缺口，並支援匯出為標準 Markdown 格式或 A4 規格 PDF 報告。")
        
        # 檢查是否有足夠的 PDF 論文
        existing_pdfs = list(DATA_DIR.glob("*.pdf"))
        
        if not gemini_api_key:
            st.info("👉 請在左側「專案控制台」輸入您的 **Gemini API Key** 以啟用文獻綜述功能。")
        elif not vector_manager or vector_manager.get_collection_count() == 0:
            st.warning("⚠️ 文獻綜述不可用：向量庫目前沒有資料！請先完成論文上傳，並在側邊欄點擊「🔄 向量化本地文獻庫」。")
        elif len(existing_pdfs) < 2:
            st.info("💡 跨文獻綜述需要至少 **2 篇** 以上的已向量化文獻。目前檢測到本地文獻庫中只有 1 篇文獻，請先前往第一分頁上傳更多文獻並點擊向量化。")
        elif not comparison_manager:
            st.error("❌ 比較引擎未啟用，無法進行分析。")
        else:
            # 提供多選選單
            pdf_names = [pdf.name for pdf in existing_pdfs]
            
            st.markdown("#### 📑 選擇您要納入綜述的學術文獻（最少 2 篇）")
            selected_review_pdfs = st.multiselect(
                label="選擇綜述文獻 PDF 檔案",
                options=pdf_names,
                default=pdf_names[:2] if len(pdf_names) >= 2 else pdf_names,
                key="review_multiselect_pdfs"
            )
            
            # 生成綜述按鈕
            submit_review = st.button("📚 啟動綜述特徵交叉比對與回顧報告生成", key="review_submit_btn")
            
            # 初始化快取
            if "review_text" not in st.session_state:
                st.session_state.review_text = None
            if "review_pdf_set" not in st.session_state:
                st.session_state.review_pdf_set = set()
                
            # 觸發生成
            if submit_review:
                if len(selected_review_pdfs) < 2:
                    st.warning("⚠️ 請至少選擇兩篇文獻以進行回顧報告生成！")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 顯示載入動畫
                    skeleton_placeholder = st.empty()
                    skeleton_placeholder.markdown(render_skeleton_screen("qa"), unsafe_allow_html=True)
                    
                    try:
                        # 1. 取得或生成特徵資料
                        comparison_data = []
                        total_pdfs = len(selected_review_pdfs)
                        
                        for idx, pdf_name in enumerate(selected_review_pdfs, 1):
                            status_text.info(f"🧬 正在從 `{pdf_name}` 提煉學術特徵 ({idx}/{total_pdfs})...")
                            features = comparison_manager.extract_features_for_paper(pdf_name)
                            comparison_data.append({
                                "pdf_file": pdf_name,
                                "title": features.title,
                                "methodology": features.methodology,
                                "datasets": features.datasets,
                                "pros": features.pros,
                                "cons": features.cons
                            })
                            progress_bar.progress(int((idx / total_pdfs) * 50))
                            
                        # 2. 初始化 LiteratureReviewGenerator 並生成報告
                        status_text.info("🤖 正在交叉比對各篇文獻並撰寫綜述報告，這可能需要一到二分鐘...")
                        from src.rag.literature_review_manager import AcademicLiteratureReviewGenerator
                        review_generator = AcademicLiteratureReviewGenerator()
                        
                        review_text = review_generator.generate_review_report(comparison_data)
                        progress_bar.progress(100)
                        
                        # 儲存快取
                        st.session_state.review_text = review_text
                        st.session_state.review_pdf_set = set(selected_review_pdfs)
                        
                        status_text.empty()
                        progress_bar.empty()
                        skeleton_placeholder.empty()
                        st.success("🎉 學術文獻回顧綜述報告生成成功！已安全緩存報告。")
                        st.balloons()
                        
                    except Exception as e:
                        status_text.empty()
                        progress_bar.empty()
                        skeleton_placeholder.empty()
                        st.error(f"❌ 綜述報告生成失敗: {e}")
                        logger.error(f"綜述報告生成失敗: {e}")
                        
            # 3. 渲染結果
            if st.session_state.review_text and set(selected_review_pdfs) == st.session_state.review_pdf_set:
                report = st.session_state.review_text
                
                st.markdown("---")
                st.markdown("#### 📚 生成的文獻綜述與研究回顧")
                
                # 渲染 Markdown 綜述報告
                st.markdown(report)
                
                # 4. 提供 PDF 下載
                st.markdown("---")
                st.markdown("#### 📥 匯出綜述報告")
                st.caption("您可以一鍵下載本篇報告的 A4 排版 PDF 檔。")
                
                try:
                    from src.utils.pdf_generator import generate_review_pdf
                    pdf_bytes = generate_review_pdf(report)
                    
                    st.download_button(
                        label="📥 下載學術綜述報告 (.pdf)",
                        data=pdf_bytes,
                        file_name="literature_review_synthesis_report.pdf",
                        mime="application/pdf",
                        key="download_review_pdf_btn",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.error(f"❌ PDF 生成失敗: {pdf_err}")

# 系統底部狀態資訊
st.markdown("---")
st.caption("Literature Reviewer - Version 1.0.0 (Gemini Edition) | Developed by Sophomore CS Students with AI Mentor")

# -*- coding: utf-8 -*-
"""
Literature Reviewer 前端 UI 樣式常數
集中管理 CSS 樣式與 HTML 橫幅範本，確保 UI 與邏輯分離。
"""

import streamlit as st

# Glassmorphism 風格與流暢動畫的 CSS 樣式
CUSTOM_CSS = """
<style>
    /* 隱藏 Streamlit 預設頂部控制列 (Deploy、三點選單、側邊欄收合鈕) */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* 調整主面板寬度與頂部間距，最大化操作空間並向兩側拉展 */
    .block-container {
        max-width: 95% !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* 頂部 API 金鑰狀態按鈕樣式 (綠色亮燈效果) */
    .api-status-container>div>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        font-size: 1rem !important;
        padding: 6px 16px !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    .api-status-container>div>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        border-color: #34d399 !important;
    }

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
    
    /* 骨架屏載入動畫 (Skeleton Screen Loader) */
    @keyframes pulse {
        0% { background-color: rgba(255, 255, 255, 0.05); }
        50% { background-color: rgba(255, 255, 255, 0.15); }
        100% { background-color: rgba(255, 255, 255, 0.05); }
    }
    .skeleton-card {
        background: rgba(30, 41, 59, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .skeleton-title {
        height: 24px;
        width: 45%;
        margin-bottom: 18px;
        border-radius: 4px;
        animation: pulse 1.5s infinite ease-in-out;
    }
    .skeleton-line {
        height: 14px;
        margin-bottom: 10px;
        border-radius: 4px;
        animation: pulse 1.5s infinite ease-in-out;
    }
</style>
"""

# 頂部 Hero Banner HTML 區塊
HERO_BANNER = """
<div class="hero-container">
    <span class="badge">Week 6 (Complete & Optimized): 評估與整合 (Evaluation & UI Integration)</span>
    <h1 class="hero-title">Literature Reviewer 📚</h1>
    <p class="hero-subtitle">大二資工系專題：基於 Google Gemini 2.5-Flash 與雙欄排版還原的學術文獻 RAG 系統</p>
</div>
"""

def render_styles() -> None:
    """渲染自訂的 CSS 樣式"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_hero_banner() -> None:
    """渲染頂部 Hero 橫幅"""
    st.markdown(HERO_BANNER, unsafe_allow_html=True)

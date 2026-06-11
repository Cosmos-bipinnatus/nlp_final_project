# -*- coding: utf-8 -*-
import time
from typing import Callable, Any
from src.utils.logger import get_logger

# 嘗試載入 streamlit，以供前端警告渲染
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# 設定日誌
logger = get_logger(__name__)

def retry_on_429(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    對 Gemini API 429 / RESOURCE_EXHAUSTED 速率限制錯誤進行重試的包裝函式。
    採用指數退避 (Exponential Backoff) 機制，最高重試 5 次，並在 Streamlit 前端顯示倒數提示。
    
    Args:
        func: 待執行的 API 呼叫函式 (例如 self.chat.invoke)
        *args: 傳遞給 func 的位置參數
        **kwargs: 傳遞給 func 的關鍵字參數
        
    Returns:
        Any: 函數執行成功後的傳回值
    """
    max_retries = 5
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            # 判斷是否為 429 限流或資源耗盡錯誤
            is_429 = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()
            
            if is_429 and attempt < max_retries - 1:
                # 指數退避計算：2, 4, 8, 16... 秒
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"[API 限流] 偵測到 429/RESOURCE_EXHAUSTED，"
                    f"將在 {delay:.1f} 秒後進行第 {attempt + 1}/{max_retries} 次重試。錯誤訊息: {err_msg}"
                )
                
                # 如果在 Streamlit 環境下，在 UI 渲染出警告，避免評審與同學慌張
                if HAS_STREAMLIT:
                    try:
                        warning_msg = (
                            f"⏳ **Gemini API 速率限制中...** "
                            f"系統將在 {delay:.1f} 秒後進行第 {attempt + 1}/{max_retries} 次自動重試，請稍候。"
                        )
                        # 避開側邊欄窄欄位被擠壓的問題：如果是由側邊欄動作觸發，將警告推送到大容器側邊欄底部
                        if st.session_state.get("sidebar_active", False):
                            st.sidebar.warning(warning_msg)
                        else:
                            st.warning(warning_msg)
                    except Exception:
                        pass
                
                time.sleep(delay)
            else:
                # 非 429 錯誤，或是已達到最大重試次數，直接拋出例外
                logger.error(f"[API 失敗] 已達重試上限或遇到非 429 錯誤: {e}")
                raise e

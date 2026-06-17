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

def set_model_on_runnable(runnable: Any, new_model_name: str) -> bool:
    """
    遞迴尋找並將 LangChain Runnable (如 RunnableBinding, ChatGoogleGenerativeAI) 中的底層模型名稱修改為 new_model_name。
    """
    updated = False
    if hasattr(runnable, "model"):
        runnable.model = new_model_name
        updated = True
    if hasattr(runnable, "model_name"):
        runnable.model_name = new_model_name
        updated = True
    if hasattr(runnable, "bound"):
        if set_model_on_runnable(runnable.bound, new_model_name):
            updated = True
    return updated

def retry_on_429(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    對 Gemini API 429 (限流) 或 503 (暫時不可用/超載) 等暫時性錯誤進行自動重試與模型自動降級自癒。
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
            # 判斷是否為 429 限流/資源耗盡，或 503 伺服器超載暫時不可用
            is_429 = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()
            is_503 = "503" in err_msg or "UNAVAILABLE" in err_msg or "overloaded" in err_msg.lower() or "not found" in err_msg.lower()
            
            # 自動自癒：若是 503 暫時不可用，或模型找不到 (例如帳號不支援 3.5)，進行自動降級
            if is_503 and attempt >= 0 and hasattr(func, "__self__"):
                obj = func.__self__
                current_model = None
                target = obj
                while target is not None:
                    current_model = getattr(target, "model", None) or getattr(target, "model_name", None)
                    if current_model:
                        break
                    target = getattr(target, "bound", None)
                    
                if current_model and "gemini-3.5-flash" in current_model:
                    fallback_model = "gemini-2.5-flash"
                    if set_model_on_runnable(obj, fallback_model):
                        logger.warning(f"⚠️ [自動降級] 成功將 {current_model} 降級為穩定版 {fallback_model}！")
                        if HAS_STREAMLIT:
                            try:
                                st.warning(f"⚠️ **Gemini 3.5 Flash 服務超載或不可用 (Error 503)**，已自動降級為穩定版 `{fallback_model}` 以確保問答與生成能正常運作。")
                            except Exception:
                                pass
            
            if (is_429 or is_503) and attempt < max_retries - 1:
                # 指數退避計算：2, 4, 8, 16... 秒
                delay = base_delay * (2 ** attempt)
                err_type_zh = "API 限流" if is_429 else "伺服器繁忙"
                logger.warning(
                    f"[{err_type_zh}] 偵測到暫時性錯誤，"
                    f"將在 {delay:.1f} 秒後進行第 {attempt + 1}/{max_retries} 次重試。錯誤訊息: {err_msg}"
                )
                
                # 如果在 Streamlit 環境下，在 UI 渲染出警告，避免使用者慌張
                if HAS_STREAMLIT:
                    try:
                        warning_msg = (
                            f"⏳ **Gemini API {err_type_zh} (Error {503 if is_503 else 429})...** "
                            f"系統將在 {delay:.1f} 秒後進行第 {attempt + 1}/{max_retries} 次自動重試，請稍候。"
                        )
                        if st.session_state.get("sidebar_active", False):
                            st.sidebar.warning(warning_msg)
                        else:
                            st.warning(warning_msg)
                    except Exception:
                        pass
                
                time.sleep(delay)
            else:
                # 非暫時性錯誤，或已達重試上限
                logger.error(f"[API 失敗] 已達重試上限或遇到非暫時性錯誤: {e}")
                raise e

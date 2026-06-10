# -*- coding: utf-8 -*-
"""
統一日誌記錄器設定
避免各子模組重複呼叫 logging.basicConfig。
"""

import logging
import sys

# 預設日誌格式
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logger(level: int = logging.INFO) -> None:
    """
    全域日誌系統初始化。
    僅在第一次呼叫時配置 root logger，避免重複呼叫造成的 handler 重複載入問題。
    """
    root_logger = logging.getLogger()
    
    # 若已經有 handler，代表已經配置過，跳過
    if root_logger.handlers:
        return
        
    root_logger.setLevel(level)
    
    # 建立標準輸出 Handler，並設定 cp950 (Windows) 的編碼容錯輸出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(DEFAULT_FORMAT)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
    取得命名日誌記錄器。
    
    Args:
        name (str): 模組名稱 (通常傳入 __name__)
        
    Returns:
        logging.Logger: 配置完成的 Logger 實體
    """
    # 確保日誌系統已初始化
    setup_logger()
    return logging.getLogger(name)

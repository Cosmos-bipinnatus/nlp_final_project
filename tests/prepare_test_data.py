# -*- coding: utf-8 -*-
"""
測試數據自動準備工具 (prepare_test_data.py)
用於自動從 ArXiv 下載經典的 Transformer 論文 (Attention Is All You Need)，
命名為測試腳本專用的 'Attentation Is All You Need.pdf' 並存入 data/ 目錄下。
"""

import sys
import urllib.request
from pathlib import Path

# 確保輸出支援 UTF-8，避免 Windows 亂碼或崩潰
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def download_test_paper() -> None:
    """自動為測試環境下載 Attention 論文數據"""
    print("==================================================")
    print(" 📥 Literature Reviewer 測試論文自動下載器")
    print("==================================================")

    # 1. 計算目標路徑
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    
    # 確保 data 目錄存在
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 測試腳本預設尋找的檔名 (注意有拼寫 Attentation)
    target_path: Path = data_dir / "Attentation Is All You Need.pdf"
    
    if target_path.exists():
        print(f"✅ 測試論文已存在，無須重複下載：\n   {target_path.relative_to(project_root)}")
        print(f"   檔案大小: {target_path.stat().st_size / 1024 / 1024:.2f} MB")
        return

    # ArXiv 上的 Transformer 論文 PDF 下載連結
    paper_url: str = "https://arxiv.org/pdf/1706.03762"
    
    print(f"正在從 ArXiv 下載經典 Transformer 論文...")
    print(f"🔗 來源連結: {paper_url}")
    print(f"💾 儲存路徑: {target_path.relative_to(project_root)}")
    
    try:
        # 模擬瀏覽器 Header 避免部分網站封鎖
        req = urllib.request.Request(
            paper_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
            # 讀取並寫入
            data = response.read()
            out_file.write(data)
            
        print("\n🎉 下載成功！")
        print(f"   已成功儲存為: {target_path.name}")
        print(f"   實體路徑: {target_path.resolve()}")
        print(f"   檔案大小: {target_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except urllib.error.URLError as url_err:
        print(f"\n❌ 下載失敗！網路連接錯誤: {url_err}")
        print("   👉 提示: 請確認您的網路連線是否通暢，或手動下載該論文後改名放入 data/ 目錄。")
    except Exception as e:
        print(f"\n❌ 下載時發生非預期錯誤: {e}")
        
    print("==================================================")

if __name__ == "__main__":
    download_test_paper()

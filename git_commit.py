import subprocess
import sys
from pathlib import Path

def run_git_commands() -> None:
    """
    使用 Python subprocess 模組自動化執行 git add 與 git commit
    這有助於學生理解如何使用 Python 與系統 CLI 工具進行整合
    """
    project_root = Path(__file__).parent.resolve()
    
    print(f"📂 專案根目錄: {project_root}")
    
    # 1. 執行 git add -A
    print("⚡ 正在執行 git add -A ...")
    add_result = subprocess.run(
        ["git", "add", "-A"],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    if add_result.returncode != 0:
        print(f"❌ git add 失敗。錯誤訊息:\n{add_result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("✅ git add 成功！")
    
    # 2. 執行 git commit
    commit_msg = (
        "Refactor: centralize configurations, unify logging system, and implement "
        "Literature Review tab with PDF exporting, Ingestion Guard, PDF Metadata heuristic extraction, "
        "and Hybrid Agent Routing."
    )
    print(f"⚡ 正在執行 git commit -m \"{commit_msg}\" ...")
    
    commit_result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    if commit_result.returncode != 0:
        # 如果沒有東西可以 commit，這不算是致命錯誤，可能是已經 commit 過了
        if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
            print("ℹ️ 沒有檢測到任何變更需要 commit。")
        else:
            print(f"❌ git commit 失敗。錯誤訊息:\n{commit_result.stderr}", file=sys.stderr)
            sys.exit(1)
    else:
        print("🎉 git commit 成功！")
        print(commit_result.stdout)

if __name__ == "__main__":
    # Windows 環境下 cp950 編碼輸出 Emoji 會出錯，強制指定以 UTF-8 輸出
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
        
    try:
        run_git_commands()
    except Exception as e:
        print(f"💥 發生非預期錯誤: {e}", file=sys.stderr)
        sys.exit(1)

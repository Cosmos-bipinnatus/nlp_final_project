# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# 將 src/ 目錄加入 Python 模組搜尋路徑
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.router_agent import AcademicRouterAgent

def test_agent_pipeline():
    print("[Info] Starting Academic Router Agent & Tools Pipeline Integration Test...")
    
    try:
        # 1. 初始化學術路由代理
        print("\n--- Step 1: Initializing AcademicRouterAgent ---")
        agent = AcademicRouterAgent(persist_directory="vectorstore")
        
        # 處理 Windows cmd/powershell cp950 編碼問題
        sys.stdout.reconfigure(errors='ignore') if hasattr(sys.stdout, 'reconfigure') else None
        
        # ---------------------------------------------
        # 測試案例一：本地 RAG 路由測試
        # ---------------------------------------------
        print("\n" + "="*80)
        print("【測試案例一：本地文獻 RAG 提問】")
        print("="*80)
        local_query = "What is the significance of Multi-Head Attention in the Transformer?"
        print(f"Student Query: '{local_query}'")
        
        print("\n[Agent Running] Analyzing and executing...")
        local_response = agent.route_and_execute(local_query)
        
        print("\n--- [Agent Decision Process] ---")
        print(f"Chosen Route: {local_response['route'].upper()}")
        print(f"Optimized Query: '{local_response['search_query']}'")
        print(f"Rationale: {local_response['rationale']}")
        
        print("\n--- [Answer Output] ---")
        print(local_response["answer"])
        print(f"Retrieved Chunks Count: {len(local_response['chunks'])}")
        
        # 驗證路由是否正確
        assert local_response["route"] == "local", "❌ 測試案例一路由錯誤，應該路由至 'local'！"
        print("\n✅ 測試案例一 (本地 RAG) 驗證成功！")
        
        # ---------------------------------------------
        # 測試案例二：外部 ArXiv 檢索測試
        # ---------------------------------------------
        print("\n" + "="*80)
        print("【測試案例二：外部學術 ArXiv 提問】")
        print("="*80)
        arxiv_query = "Please search for the latest papers on RAG architecture on arXiv."
        print(f"Student Query: '{arxiv_query}'")
        
        print("\n[Agent Running] Analyzing and executing...")
        arxiv_response = agent.route_and_execute(arxiv_query)
        
        print("\n--- [Agent Decision Process] ---")
        print(f"Chosen Route: {arxiv_response['route'].upper()}")
        print(f"Optimized Query: '{arxiv_response['search_query']}'")
        print(f"Rationale: {arxiv_response['rationale']}")
        
        print("\n--- [Compiled Research Report Output] ---")
        print(arxiv_response["answer"])
        print(f"Retrieved Papers Metadata Count: {len(arxiv_response['papers'])}")
        
        # 驗證路由是否正確
        assert arxiv_response["route"] == "arxiv", "❌ 測試案例二路由錯誤，應該路由至 'arxiv'！"
        print("\n✅ 測試案例二 (外部 ArXiv) 驗證成功！")
        
        print("\n" + "="*80)
        print("🎉 [Success] All Week 5 Router Agent & Tools Pipeline Integration Tests Completed Successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ [Error] Agent pipeline test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_agent_pipeline()

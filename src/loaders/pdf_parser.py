import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DoubleColumnPDFParser:
    """
    學術論文雙欄排版解析器。
    使用 PyMuPDF (fitz) 提取文本與坐標，並利用「水平屏障雙欄排序演算法」
    完美還原閱讀順序，同時提取頁碼與檔名等 Metadata 以利 Citation 引用標記。
    """
    
    def __init__(self, file_path: str | Path):
        """
        初始化解析器。
        
        Args:
            file_path (str | Path): PDF 檔案的路徑
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"找不到指定的 PDF 檔案: {self.file_path}")
            
    def _is_full_width_barrier(self, block: Tuple[float, float, float, float, str, int, int], page_width: float) -> bool:
        """
        判斷一個文本區塊是否為「跨欄屏障」（例如：大標題、全幅摘要、跨頁圖表或表格）。
        
        運作機制：
        如果區塊的左邊界小於頁面的 40%，且右邊界大於頁面的 60%，就認定它橫跨了左右兩欄。
        """
        x0, _, x1, _, _, _, _ = block
        mid_left_boundary = page_width * 0.40
        mid_right_boundary = page_width * 0.60
        
        # 橫跨了中央分界線的區塊，視為水平屏障
        return x0 < mid_left_boundary and x1 > mid_right_boundary

    def _sort_blocks_by_reading_order(self, blocks: List[Tuple[float, float, float, float, str, int, int]], page_width: float) -> str:
        """
        核心演算法：水平屏障雙欄排序演算法 (Horizontal Barrier Double-Column Sorting)
        
        將頁面中的文本區塊依照正確的閱讀順序（先左欄由上至下，再右欄由上至下，跨欄區塊居中）進行重組。
        """
        mid_x = page_width / 2
        
        # 1. 過濾非文字區塊 (block_type == 0 代表文字，1 代表圖片或其它)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        
        # 2. 將所有區塊依垂直坐標 y0 由上至下排序
        sorted_by_y = sorted(text_blocks, key=lambda b: b[1])
        
        reconstructed_text_segments = []
        current_section_blocks = []
        
        for block in sorted_by_y:
            # 判斷是否為跨欄的「水平屏障」
            if self._is_full_width_barrier(block, page_width):
                # 遇到屏障！先處理累積在當前區域的雙欄文字
                if current_section_blocks:
                    reconstructed_text_segments.append(
                        self._process_double_column_section(current_section_blocks, mid_x)
                    )
                    current_section_blocks = []
                
                # 直接添加跨欄屏障的文字，並在其前後加上換行
                reconstructed_text_segments.append(block[4].strip())
            else:
                # 屬於一般的欄位文字，先累積起來
                current_section_blocks.append(block)
                
        # 處理剩餘的最後一個區域
        if current_section_blocks:
            reconstructed_text_segments.append(
                self._process_double_column_section(current_section_blocks, mid_x)
            )
            
        # 用空行連接重組後的各個段落
        return "\n\n".join(reconstructed_text_segments)

    def _process_double_column_section(self, blocks: List[Tuple[float, float, float, float, str, int, int]], mid_x: float) -> str:
        """
        處理一個垂直區間內的雙欄文字排序。
        在此區間內，區塊會被分流為「左欄」與「右欄」，分別由上至下排序，最後再進行左欄接右欄拼接。
        """
        left_column = []
        right_column = []
        
        for b in blocks:
            x0, y0, x1, y1, text, _, _ = b
            # 計算區塊的中心點 x 坐標
            center_x = (x0 + x1) / 2
            
            # 若中心點偏左，歸入左欄；否則歸入右欄
            if center_x < mid_x:
                left_column.append(b)
            else:
                right_column.append(b)
                
        # 左右欄分別依 y0 (頂部垂直坐標) 進行排序
        left_sorted = sorted(left_column, key=lambda b: b[1])
        right_sorted = sorted(right_column, key=lambda b: b[1])
        
        # 拼接文字：先讀完左欄，再讀右欄
        left_text = "\n".join([b[4].strip() for b in left_sorted])
        right_text = "\n".join([b[4].strip() for b in right_sorted])
        
        if left_text and right_text:
            return f"{left_text}\n{right_text}"
        return left_text if left_text else right_text

    def parse_pdf(self) -> List[Dict[str, Any]]:
        """
        解析整個 PDF 檔案。
        針對每一頁，還原雙欄閱讀順序，並打包成包含 Page Content 與 Metadata 的字典清單。
        
        Returns:
            List[Dict[str, Any]]: 包含解析後文本與 metadata 的 dict 清單。
            格式：
            [
                {
                    "page_content": "解析後的重組文本內容...",
                    "metadata": {
                        "source": "論文檔名.pdf",
                        "page": 1,
                        "total_pages": 12
                    }
                },
                ...
            ]
        """
        documents = []
        file_name = self.file_path.name
        
        try:
            # 開啟 PDF 文件
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            logger.info(f"開始解析 PDF: {file_name}，共 {total_pages} 頁")
            
            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_number = page_idx + 1  # 頁碼採 1-based 表示
                
                # 獲取頁面尺寸，用以計算中線
                page_width = page.rect.width
                
                # 獲取詳細的區塊坐標與文本資訊
                # get_text("blocks") 回傳格式：
                # (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
                
                # 執行雙欄還原演算法
                reconstructed_text = self._sort_blocks_by_reading_order(blocks, page_width)
                
                # 包裝 Metadata
                metadata = {
                    "source": file_name,
                    "page": page_number,
                    "total_pages": total_pages
                }
                
                documents.append({
                    "page_content": reconstructed_text,
                    "metadata": metadata
                })
                
            doc.close()
            logger.info(f"PDF 解析完成: {file_name}")
            return documents
            
        except Exception as e:
            logger.error(f"解析 PDF 檔案 {file_name} 時發生異常: {e}")
            raise e

# ==========================================
# 5. 單元測試與驗證模組
# ==========================================
if __name__ == "__main__":
    # 這段程式碼在直接執行此腳本時會運行，用以驗證解析管線是否正常
    test_pdf_path = Path("data") / "test_paper.pdf"
    
    # 建立一個測試用的空 PDF（如果不存在）或讓學生上傳後測試
    if not test_pdf_path.exists():
        print(f"💡 提示：請先上傳 PDF 檔案至 {test_pdf_path.parent}，並命名為 test_paper.pdf 以進行解析測試。")
    else:
        print(f"🔍 開始測試雙欄 PDF 解析：{test_pdf_path}")
        parser = DoubleColumnPDFParser(test_pdf_path)
        try:
            results = parser.parse_pdf()
            print("\n=== 第一頁解析內容範例 (前 300 字) ===")
            print(results[0]["page_content"][:300] + "...")
            print("\n=== 提取的 Metadata 範例 ===")
            print(results[0]["metadata"])
            print("\n✅ 雙欄解析測試成功！")
        except Exception as e:
            print(f"❌ 測試失敗: {e}")

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Tuple
from src.utils.logger import get_logger

# 設定日誌
logger = get_logger(__name__)

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

    def _extract_metadata_heuristics(self, doc) -> Dict[str, str]:
        """
        利用啟發式演算法從 PDF 的第一頁提取標題 (Title)、作者 (Authors) 與摘要 (Abstract)。 (B3)
        
        Args:
            doc: PyMuPDF Document 物件
            
        Returns:
            Dict[str, str]: 包含 title, authors, abstract 的字典
        """
        metadata = {
            "title": "",
            "authors": "",
            "abstract": ""
        }
        
        try:
            if len(doc) == 0:
                return metadata
                
            # 取得第一頁
            page = doc[0]
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])
            
            # 1. 蒐集所有 spans 資訊
            spans_info = []
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            text = s["text"].strip()
                            if text:
                                spans_info.append({
                                    "text": text,
                                    "size": s["size"],
                                    "bbox": s["bbox"]
                                })
                                
            if not spans_info:
                return metadata
                
            # 2. 提取標題：通常是字型最大的一行或多行
            # 過濾掉長度小於等於 3 的雜訊
            valid_spans = [s for s in spans_info if len(s["text"]) > 3]
            if not valid_spans:
                return metadata
                
            max_size = max(s["size"] for s in valid_spans)
            # 提取大於等於最大字型 90% 的 spans (多行標題)
            title_spans = [s for s in valid_spans if s["size"] >= max_size * 0.9]
            # 依垂直坐標 y0 排序，若相同則依 x0 排序
            title_spans = sorted(title_spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
            
            title = " ".join(s["text"] for s in title_spans)
            title = " ".join(title.split()) # 清理連續空格
            metadata["title"] = title
            
            # 計算標題的底部邊界 y1
            title_bottom_y = max(s["bbox"][3] for s in title_spans)
            
            # 3. 提取作者：通常在標題下方，且字型大於普通內文，小於標題
            # 過濾出在標題下方的 spans
            below_title_spans = [s for s in valid_spans if s["bbox"][1] > title_bottom_y]
            
            # 通常作者在 "Abstract" 或 "ABSTRACT" 之前出現
            abstract_y = None
            for s in below_title_spans:
                if s["text"].lower() in ["abstract", "abstract—", "abstract:"]:
                    abstract_y = s["bbox"][1]
                    break
                    
            if abstract_y is None:
                # 模糊搜尋 abstract 關鍵字
                for s in below_title_spans:
                    if "abstract" in s["text"].lower():
                        abstract_y = s["bbox"][1]
                        break
                        
            # 如果有找到 Abstract 的 y 軸位置，則作者 span 應在 title_bottom_y 與 abstract_y 之間
            author_candidates = []
            if abstract_y is not None:
                author_candidates = [s for s in below_title_spans if s["bbox"][3] < abstract_y]
            else:
                # 否則只取標題下方前幾個 spans
                author_candidates = below_title_spans[:10]
                
            # 作者的字型通常相同，且通常在一個或兩個連續 block 中
            if author_candidates:
                # 過濾出字型大小在合理區區間 (8pt ~ 14pt) 的 span，且長度符合名字特性
                # 剔除可能包含 email 或機構的 span
                clean_authors = []
                for s in author_candidates:
                    txt = s["text"]
                    if not any(word in txt.lower() for word in ["@", "email", "university", "department", "inst", "school", "college"]):
                        clean_authors.append(s)
                
                # 依坐標排序
                clean_authors = sorted(clean_authors, key=lambda s: (s["bbox"][1], s["bbox"][0]))
                authors_str = " ".join(s["text"] for s in clean_authors)
                authors_str = " ".join(authors_str.split())
                # 限制長度
                if len(authors_str) > 150:
                    authors_str = authors_str[:150] + "..."
                metadata["authors"] = authors_str
            else:
                metadata["authors"] = "未知作者"
            
            # 4. 提取摘要：通常在 "Abstract" 關鍵字後面的文字區塊，或是包覆在 "Abstract" 與 "Introduction" 之間
            raw_blocks = page.get_text("blocks")
            raw_blocks = sorted(raw_blocks, key=lambda b: (b[1], b[0]))
            
            abstract_text = ""
            found_abstract = False
            for b in raw_blocks:
                if b[6] == 0: # 文字區塊
                    text = b[4].strip()
                    if not found_abstract:
                        # 偵測是否為 Abstract 開頭
                        if text.lower().startswith("abstract") or "abstract" in text[:30].lower():
                            found_abstract = True
                            # 取出 Abstract 字眼後面的內容
                            idx = text.lower().find("abstract")
                            part = text[idx + 8:].strip()
                            if part.startswith("—") or part.startswith("-") or part.startswith(":"):
                                part = part[1:].strip()
                            abstract_text = part
                    else:
                        # 結束條件：通常 Abstract 後面是 Introduction
                        if "introduction" in text[:25].lower() or "1. intro" in text[:25].lower() or "i. intro" in text[:25].lower():
                            break
                        abstract_text += "\n" + text
                        
            if abstract_text:
                abstract_clean = " ".join(abstract_text.split())
                if len(abstract_clean) > 800:
                    abstract_clean = abstract_clean[:800] + "..."
                metadata["abstract"] = abstract_clean
            else:
                metadata["abstract"] = "無摘要資料"
                
        except Exception as e:
            logger.warning(f"啟發式提取 PDF 後設資料失敗: {e}")
            
        return metadata

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
                        "total_pages": 12,
                        "title": "提取之實際標題",
                        "authors": "提取之作者",
                        "abstract": "提取之摘要"
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
            
            # 提取後設資料 (B3)
            paper_meta = self._extract_metadata_heuristics(doc)
            title = paper_meta.get("title") or file_name
            authors = paper_meta.get("authors") or "未知作者"
            abstract = paper_meta.get("abstract") or "無摘要資料"
            logger.info(f"成功提取文獻後設資料 -> 標題: '{title}' | 作者: '{authors}'")
            
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
                
                # 包裝 Metadata，加入標題、作者和摘要 (B3)
                metadata = {
                    "source": file_name,
                    "page": page_number,
                    "total_pages": total_pages,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract
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

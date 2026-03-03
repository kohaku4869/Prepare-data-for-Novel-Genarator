"""
File Reader Module
Đọc file text (.txt) hoặc PDF (.pdf) và trả về nội dung dạng string.
"""

import os
import logging

logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """
    Đọc file và trả về nội dung dạng string.
    Tự động detect loại file dựa vào extension.

    Args:
        file_path: Đường dẫn đến file cần đọc.

    Returns:
        Nội dung file dạng string.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu định dạng file không được hỗ trợ.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return _read_text_file(file_path)
    elif ext == ".pdf":
        return _read_pdf_file(file_path)
    else:
        raise ValueError(
            f"Định dạng file không được hỗ trợ: {ext}. "
            "Chỉ hỗ trợ .txt và .pdf"
        )


def _read_text_file(file_path: str) -> str:
    """Đọc file text UTF-8."""
    logger.info(f"Đọc file text: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"Đọc xong: {len(content)} ký tự")
    return content


def _read_pdf_file(file_path: str) -> str:
    """Đọc file PDF dùng PyMuPDF (fitz)."""
    import fitz  # pymupdf

    logger.info(f"Đọc file PDF: {file_path}")
    doc = fitz.open(file_path)
    num_pages = len(doc)
    pages = []
    for page_num in range(num_pages):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()

    content = "\n\n".join(pages)
    logger.info(f"Đọc xong: {num_pages} trang, {len(content)} ký tự")
    return content

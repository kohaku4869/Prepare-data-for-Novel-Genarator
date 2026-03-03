"""
Text Splitter Module
Tách văn bản thành các chapter/đoạn văn và chia nhỏ nếu quá dài.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Regex patterns để detect chapter headings
CHAPTER_PATTERNS = [
    # English: Chapter 1, Chapter I, CHAPTER ONE
    r"(?:^|\n)(\s*(?:Chapter|CHAPTER)\s+[\dIVXLCDMivxlcdm]+[^\n]*)",
    # Vietnamese: Chương 1, CHƯƠNG I, Phần 1, PHẦN I
    r"(?:^|\n)(\s*(?:Chương|CHƯƠNG|Phần|PHẦN|Hồi|HỒI)\s+[\dIVXLCDMivxlcdm]+[^\n]*)",
    # Separator patterns: ***, ---, ===
    r"(?:^|\n)(\s*[*\-=]{3,}\s*(?:\n|$))",
    # Numbered headings: 1., 2., etc. at start of line (with title text)
    r"(?:^|\n)(\s*\d+\.\s+[A-ZÀ-Ỹ][^\n]{3,})",
]


@dataclass
class TextChunk:
    """Một đoạn văn bản đã được chia."""
    chunk_id: str
    text: str
    chapter: str | None = None


def split_text(
    text: str,
    max_chunk_size: int = 4000,
    min_chunk_size: int = 100,
) -> list[TextChunk]:
    """
    Tách văn bản thành các chunk phù hợp.

    Args:
        text: Văn bản gốc.
        max_chunk_size: Kích thước chunk tối đa (ký tự).
        min_chunk_size: Kích thước chunk tối thiểu (ký tự).

    Returns:
        Danh sách TextChunk.
    """
    # Bước 1: Thử tách theo chapter
    chapters = _split_by_chapters(text)

    if len(chapters) <= 1:
        # Không tìm thấy chapter, tách theo đoạn văn
        logger.info("Không tìm thấy chapter heading, tách theo đoạn văn")
        chapters = [("Toàn bộ văn bản", text)]

    # Bước 2: Với mỗi chapter, chia nhỏ nếu quá dài
    chunks = []
    global_idx = 0

    for chapter_name, chapter_text in chapters:
        chapter_text = chapter_text.strip()
        if len(chapter_text) < min_chunk_size:
            continue

        if len(chapter_text) <= max_chunk_size:
            chunks.append(TextChunk(
                chunk_id=f"chunk_{global_idx:04d}",
                text=chapter_text,
                chapter=chapter_name,
            ))
            global_idx += 1
        else:
            # Chia nhỏ chapter
            sub_chunks = _split_long_text(chapter_text, max_chunk_size, min_chunk_size)
            for i, sub_text in enumerate(sub_chunks):
                chunks.append(TextChunk(
                    chunk_id=f"chunk_{global_idx:04d}",
                    text=sub_text,
                    chapter=f"{chapter_name} (phần {i + 1}/{len(sub_chunks)})",
                ))
                global_idx += 1

    logger.info(f"Tách xong: {len(chunks)} chunks từ {len(chapters)} chapters")
    return chunks


def _split_by_chapters(text: str) -> list[tuple[str, str]]:
    """
    Tách văn bản theo chapter headings.
    Returns: list of (chapter_name, chapter_text)
    """
    # Tìm tất cả vị trí chapter heading
    split_positions = []

    for pattern in CHAPTER_PATTERNS:
        for match in re.finditer(pattern, text):
            heading = match.group(1).strip()
            # Bỏ qua separator patterns cho tên chapter
            if re.match(r"^[*\-=]+$", heading):
                heading = None
            start = match.start()
            split_positions.append((start, heading))

    if not split_positions:
        return []

    # Sắp xếp theo vị trí
    split_positions.sort(key=lambda x: x[0])

    # Loại bỏ duplicates gần nhau (trong phạm vi 10 ký tự)
    filtered = [split_positions[0]]
    for pos, heading in split_positions[1:]:
        if pos - filtered[-1][0] > 10:
            filtered.append((pos, heading))
    split_positions = filtered

    # Nếu chỉ tìm thấy 1 heading, không tách
    if len(split_positions) < 2:
        return []

    # Tạo chapters
    chapters = []

    # Text trước chapter đầu tiên (nếu có)
    first_pos = split_positions[0][0]
    if first_pos > 100:  # Có nội dung đáng kể trước chapter đầu
        chapters.append(("Mở đầu", text[:first_pos]))

    for i, (pos, heading) in enumerate(split_positions):
        # Xác định end position
        if i + 1 < len(split_positions):
            end_pos = split_positions[i + 1][0]
        else:
            end_pos = len(text)

        chapter_text = text[pos:end_pos]
        chapter_name = heading if heading else f"Phần {len(chapters) + 1}"
        chapters.append((chapter_name, chapter_text))

    return chapters


def _split_long_text(
    text: str,
    max_size: int,
    min_size: int,
) -> list[str]:
    """
    Chia text dài thành các phần nhỏ hơn max_size.
    Ưu tiên chia theo đoạn văn, nếu không được thì theo câu.
    """
    # Thử chia theo đoạn văn (double newline)
    paragraphs = re.split(r"\n\s*\n", text)

    if len(paragraphs) > 1:
        return _merge_into_chunks(paragraphs, max_size, min_size, separator="\n\n")

    # Nếu chỉ có 1 đoạn, chia theo câu
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    if len(sentences) > 1:
        return _merge_into_chunks(sentences, max_size, min_size, separator=" ")

    # Fallback: chia theo ký tự (giữ nguyên ranh giới từ)
    return _split_by_words(text, max_size)


def _merge_into_chunks(
    parts: list[str],
    max_size: int,
    min_size: int,
    separator: str,
) -> list[str]:
    """
    Gộp các phần nhỏ thành chunks sao cho mỗi chunk <= max_size.
    """
    chunks = []
    current_parts = []
    current_size = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_size = len(part) + len(separator)

        if current_size + part_size > max_size and current_parts:
            chunks.append(separator.join(current_parts))
            current_parts = []
            current_size = 0

        current_parts.append(part)
        current_size += part_size

    if current_parts:
        last_chunk = separator.join(current_parts)
        # Nếu chunk cuối quá nhỏ, gộp vào chunk trước
        if len(last_chunk) < min_size and chunks:
            chunks[-1] = chunks[-1] + separator + last_chunk
        else:
            chunks.append(last_chunk)

    return chunks


def _split_by_words(text: str, max_size: int) -> list[str]:
    """Fallback: chia theo từ khi không thể chia theo câu/đoạn."""
    words = text.split()
    chunks = []
    current_words = []
    current_size = 0

    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = []
            current_size = 0
        current_words.append(word)
        current_size += word_size

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

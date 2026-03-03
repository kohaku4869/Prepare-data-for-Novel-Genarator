"""
Text Matcher Module
Dùng regex tìm đoạn văn gốc trong văn bản dựa trên start_index và end_index.
"""

import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


def find_text_in_original(
    original_text: str,
    start_index: str,
    end_index: str,
) -> str | None:
    """
    Tìm đoạn văn trong văn bản gốc dựa trên start_index và end_index.

    Args:
        original_text: Văn bản gốc đầy đủ.
        start_index: 5-7 từ đầu tiên của đoạn cần tìm.
        end_index: 5-7 từ cuối cùng của đoạn cần tìm.

    Returns:
        Đoạn văn tìm được, hoặc None nếu không tìm thấy.
    """
    # Thử exact match trước
    result = _exact_match(original_text, start_index, end_index)
    if result:
        return result

    # Thử fuzzy match (normalize whitespace)
    result = _fuzzy_match(original_text, start_index, end_index)
    if result:
        logger.info("Tìm thấy bằng fuzzy match (normalized whitespace)")
        return result

    # Thử match từng phần
    result = _partial_match(original_text, start_index, end_index)
    if result:
        logger.info("Tìm thấy bằng partial match")
        return result

    logger.warning(
        f"Không tìm thấy đoạn văn. "
        f"start='{start_index[:30]}...', end='...{end_index[-30:]}'"
    )
    return None


def _exact_match(text: str, start: str, end: str) -> str | None:
    """Tìm kiếm exact match."""
    start_escaped = re.escape(start)
    end_escaped = re.escape(end)

    pattern = f"{start_escaped}(.*?){end_escaped}"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Trả về toàn bộ: start + middle + end
        full_start = match.start()
        full_end = match.end()
        return text[full_start:full_end]

    return None


def _fuzzy_match(text: str, start: str, end: str) -> str | None:
    """
    Fuzzy match: normalize whitespace và unicode trước khi tìm.
    """
    norm_text = _normalize_text(text)
    norm_start = _normalize_text(start)
    norm_end = _normalize_text(end)

    # Tìm vị trí trong normalized text
    start_escaped = re.escape(norm_start)
    end_escaped = re.escape(norm_end)

    pattern = f"{start_escaped}(.*?){end_escaped}"
    match = re.search(pattern, norm_text, re.DOTALL)

    if match:
        # Map vị trí từ normalized text về original text
        # Dùng cách tiếp cận: tìm lại trong original dùng relaxed pattern
        return _relaxed_search(text, start, end)

    return None


def _partial_match(text: str, start: str, end: str) -> str | None:
    """
    Thử match với phần ngắn hơn của start/end index.
    Giảm dần từ full → 4 từ → 3 từ.
    """
    start_words = start.split()
    end_words = end.split()

    # Thử giảm dần số từ
    for n_words in range(min(len(start_words), len(end_words)), 2, -1):
        short_start = " ".join(start_words[:n_words])
        short_end = " ".join(end_words[-n_words:])

        result = _relaxed_search(text, short_start, short_end)
        if result:
            return result

    return None


def _relaxed_search(text: str, start: str, end: str) -> str | None:
    """
    Tìm kiếm với whitespace linh hoạt (cho phép \n, \r, multiple spaces).
    """
    # Tạo pattern cho phép bất kỳ whitespace nào giữa các từ
    start_pattern = r"\s+".join(re.escape(w) for w in start.split())
    end_pattern = r"\s+".join(re.escape(w) for w in end.split())

    pattern = f"({start_pattern})(.*?)({end_pattern})"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(0)

    return None


def _normalize_text(text: str) -> str:
    """Normalize text: unicode NFC + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

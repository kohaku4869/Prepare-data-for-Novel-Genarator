"""
LLM Client Module
Gọi Gemini API để sinh instruction_prompt cho mỗi đoạn văn.
"""

import json
import time
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Bạn là một chuyên gia về xử lý ngôn ngữ tự nhiên và huấn luyện mô hình ngôn ngữ.

Nhiệm vụ của bạn: Cho một đoạn văn trích từ tiểu thuyết, bạn cần tạo ra:

1. **instruction_prompt**: Một câu lệnh (prompt) bằng tiếng Việt, chi tiết và cụ thể, sao cho nếu đưa câu lệnh này cho một mô hình ngôn ngữ (LLM), mô hình đó sẽ viết ra một đoạn văn có:
   - Cùng NỘI DUNG (cốt truyện, sự kiện, nhân vật, hành động)
   - Cùng PHONG CÁCH viết (giọng văn, cách dùng từ, nhịp điệu)
   - Cùng CẢM XÚC và KHÔNG KHÍ
   - Cùng NGÔI KỂ và ĐIỂM NHÌN
   Câu lệnh phải đủ chi tiết để model tái tạo đoạn văn chính xác nhất có thể.

2. **start_index**: Copy nguyên văn 5 đến 7 từ ĐẦU TIÊN của đoạn văn (giữ nguyên dấu câu, viết hoa/thường).

3. **end_index**: Copy nguyên văn 5 đến 7 từ CUỐI CÙNG của đoạn văn (giữ nguyên dấu câu, viết hoa/thường).

Trả về ĐÚNG định dạng JSON sau (không thêm bất kỳ text nào khác):
{
  "instruction_prompt": "...",
  "start_index": "...",
  "end_index": "..."
}

LƯU Ý QUAN TRỌNG:
- instruction_prompt phải cụ thể về nội dung, KHÔNG ĐƯỢC chung chung kiểu "viết một đoạn văn hay"
- instruction_prompt phải mô tả chính xác các sự kiện, nhân vật, cảm xúc trong đoạn văn
- start_index và end_index PHẢI được copy chính xác từ đoạn văn gốc, KHÔNG ĐƯỢC thay đổi hay viết lại
- Chỉ trả về JSON, không có markdown code block hay text thừa\
"""


def create_client(api_key: str) -> genai.Client:
    """Tạo Gemini API client."""
    return genai.Client(api_key=api_key)


def generate_instruction(
    client: genai.Client,
    chunk_text: str,
    model: str = "gemini-2.0-flash",
    max_retries: int = 10,
    delay: float = 4.0,
) -> dict | None:
    """
    Gọi Gemini API để sinh instruction_prompt cho một đoạn văn.

    Args:
        client: Gemini API client.
        chunk_text: Đoạn văn cần xử lý.
        model: Tên model Gemini.
        max_retries: Số lần retry tối đa.
        delay: Thời gian chờ giữa các request (giây).

    Returns:
        Dict với keys: instruction_prompt, start_index, end_index.
        None nếu thất bại sau max_retries lần.
    """
    user_message = f"Đoạn văn cần xử lý:\n\n---\n{chunk_text}\n---"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )

            result = _parse_response(response.text)
            if result:
                logger.debug(f"Thành công (lần {attempt})")
                time.sleep(delay)
                return result
            else:
                logger.warning(
                    f"Lần {attempt}/{max_retries}: Không parse được response: "
                    f"{response.text[:200]}"
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Lần {attempt}/{max_retries}: JSON decode error: {e}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f"Lần {attempt}/{max_retries}: API error: {e}")
            # Nếu rate limit, chờ lâu hơn
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = min(10 * (2 ** (attempt - 1)), 320)
                logger.info(f"Rate limited, chờ {wait_time}s...")
                time.sleep(wait_time)
                continue

        if attempt < max_retries:
            time.sleep(delay)

    logger.error(f"Thất bại sau {max_retries} lần thử")
    return None


def _parse_response(response_text: str) -> dict | None:
    """
    Parse JSON response từ Gemini.
    Xử lý cả trường hợp response có markdown code block.
    """
    text = response_text.strip()

    # Loại bỏ markdown code block nếu có
    if text.startswith("```"):
        # Tìm dòng đầu tiên và dòng cuối cùng
        lines = text.split("\n")
        # Bỏ dòng ``` đầu và cuối
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    # Validate required fields
    required = ["instruction_prompt", "start_index", "end_index"]
    if not all(key in data for key in required):
        logger.warning(f"Thiếu field bắt buộc. Có: {list(data.keys())}")
        return None

    # Validate non-empty
    if not all(data[key].strip() for key in required):
        logger.warning("Có field rỗng trong response")
        return None

    return {
        "instruction_prompt": data["instruction_prompt"].strip(),
        "start_index": data["start_index"].strip(),
        "end_index": data["end_index"].strip(),
    }

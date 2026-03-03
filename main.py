"""
Novel Processing Pipeline
Đọc tiểu thuyết → tách chapter → gọi Gemini API → sinh JSONL training data.

Usage:
    python main.py --input novel.txt --output output.jsonl
    python main.py --input novel.pdf --output output.jsonl --max-chunk-size 3000
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

from file_reader import read_file
from text_splitter import split_text
from llm_client import create_client, generate_instruction
from text_matcher import find_text_in_original
from output_writer import JSONLWriter, OutputRecord

load_dotenv()


def setup_logging(verbose: bool = False):
    """Cấu hình logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Tạo instruction training data từ tiểu thuyết",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --input novel.txt --output output.jsonl
  python main.py --input novel.pdf --api-key YOUR_KEY --max-chunk-size 3000
  python main.py --input novel.txt --delay 2 --verbose
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Đường dẫn file đầu vào (txt hoặc pdf)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.jsonl",
        help="Đường dẫn file JSONL đầu ra (mặc định: output.jsonl)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4000,
        help="Kích thước chunk tối đa (ký tự, mặc định: 4000)",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Kích thước chunk tối thiểu (ký tự, mặc định: 100)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (hoặc dùng biến GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Model Gemini (mặc định: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Delay giữa các API call (giây, mặc định: 4.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Số lần retry tối đa cho mỗi API call (mặc định: 10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiện log chi tiết (debug mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("pipeline")

    # === 1. Validate API key ===
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Thiếu API key! Dùng --api-key hoặc set GEMINI_API_KEY trong .env")
        sys.exit(1)

    # === 2. Đọc file ===
    logger.info(f"📖 Đọc file: {args.input}")
    try:
        raw_text = read_file(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Lỗi đọc file: {e}")
        sys.exit(1)
    logger.info(f"✅ Đọc xong: {len(raw_text)} ký tự")

    # === 3. Tách text thành chunks ===
    logger.info(f"✂️  Tách text (max_chunk={args.max_chunk_size})...")
    chunks = split_text(
        raw_text,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
    )
    logger.info(f"✅ Tách xong: {len(chunks)} chunks")

    if not chunks:
        print("❌ Không tách được chunk nào từ file!")
        sys.exit(1)

    # === 4. Load các chunk đã xử lý (resume support) ===
    done_chunk_ids = _load_done_chunks(args.output)
    if done_chunk_ids:
        logger.info(f"♻️  Resume: đã có {len(done_chunk_ids)} chunks, bỏ qua")

    # === 5. Xử lý từng chunk ===
    logger.info(f"🤖 Gọi Gemini API ({args.model})...")
    client = create_client(api_key)

    success_count = len(done_chunk_ids)
    fail_count = 0
    skip_count = 0

    try:
        with JSONLWriter(args.output, append=True) as writer:
            remaining = [c for c in chunks if c.chunk_id not in done_chunk_ids]
            for chunk in tqdm(remaining, desc="Processing", unit="chunk",
                              initial=len(done_chunk_ids), total=len(chunks)):
                # Gọi LLM
                result = generate_instruction(
                    client=client,
                    chunk_text=chunk.text,
                    model=args.model,
                    max_retries=args.max_retries,
                    delay=args.delay,
                )

                if result is None:
                    logger.warning(f"⚠️  Skip {chunk.chunk_id}: LLM thất bại")
                    fail_count += 1
                    continue

                # Regex match tìm đoạn văn gốc
                matched_text = find_text_in_original(
                    original_text=raw_text,
                    start_index=result["start_index"],
                    end_index=result["end_index"],
                )

                if matched_text is None:
                    # Fallback: dùng chunk text gốc
                    logger.warning(
                        f"⚠️  {chunk.chunk_id}: Không match được regex, "
                        f"dùng chunk text gốc"
                    )
                    matched_text = chunk.text
                    skip_count += 1

                # Ghi JSONL ngay lập tức
                record = OutputRecord(
                    instruction=result["instruction_prompt"],
                    output=matched_text,
                    chunk_id=chunk.chunk_id,
                    chapter=chunk.chapter,
                )
                writer.write_record(record)
                success_count += 1

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Đã dừng! Dữ liệu đã ghi được lưu trong {args.output}")
        print(f"   Chạy lại lệnh cũ để tiếp tục từ chunk tiếp theo.")

    # === 6. Báo cáo kết quả ===
    print(f"\n{'='*50}")
    print(f"📊 KẾT QUẢ:")
    print(f"   Tổng chunks:     {len(chunks)}")
    print(f"   ✅ Thành công:    {success_count}")
    print(f"   ⚠️  Fallback:     {skip_count}")
    print(f"   ❌ Thất bại:      {fail_count}")
    print(f"   📄 Output file:   {args.output}")
    print(f"{'='*50}")


def _load_done_chunks(output_path: str) -> set[str]:
    """Đọc file output đã có để lấy danh sách chunk_id đã xử lý."""
    done = set()
    if not os.path.exists(output_path):
        return done
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "chunk_id" in data:
                    done.add(data["chunk_id"])
    except Exception:
        pass
    return done


if __name__ == "__main__":
    main()

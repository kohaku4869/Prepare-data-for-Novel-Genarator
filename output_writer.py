"""
Output Writer Module
Ghi kết quả ra file JSONL (mỗi dòng một JSON object).
"""

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OutputRecord:
    """Một record trong file JSONL output."""
    instruction: str
    output: str
    chunk_id: str | None = None
    chapter: str | None = None


class JSONLWriter:
    """Writer ghi JSONL theo append mode."""

    def __init__(self, output_path: str, append: bool = False):
        self.output_path = output_path
        self.count = 0
        self._mode = "a" if append else "w"

    def __enter__(self):
        self.file = open(self.output_path, self._mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        logger.info(f"Đã ghi {self.count} records vào {self.output_path}")

    def write_record(self, record: OutputRecord):
        """Ghi một record vào file."""
        data = {
            "instruction": record.instruction,
            "output": record.output,
        }
        # Thêm metadata nếu có
        if record.chunk_id:
            data["chunk_id"] = record.chunk_id
        if record.chapter:
            data["chapter"] = record.chapter

        line = json.dumps(data, ensure_ascii=False)
        self.file.write(line + "\n")
        self.file.flush()
        self.count += 1

    def write_many(self, records: list[OutputRecord]):
        """Ghi nhiều records."""
        for record in records:
            self.write_record(record)

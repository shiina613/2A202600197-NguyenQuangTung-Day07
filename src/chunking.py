from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Trường hợp đầu vào rỗng hoặc chỉ có khoảng trắng: không có chunk nào.
        if not text or not text.strip():
            return []

        # Dùng regex để tách câu theo các dấu kết thúc phổ biến (. ! ?)
        # kết hợp khoảng trắng/newline phía sau dấu câu.
        # Ví dụ: "Hello. World!" -> ["Hello.", "World!"]
        raw_sentences = re.split(r"(?<=[.!?])(?:\s+|\n+)", text.strip())

        # Làm sạch dữ liệu sau tách:
        # - loại bỏ khoảng trắng đầu/cuối mỗi câu
        # - bỏ các phần rỗng do nhiều khoảng trắng/newline liên tiếp
        sentences = [s.strip() for s in raw_sentences if s and s.strip()]

        # Nếu regex không tách được (ví dụ văn bản không có dấu câu),
        # toàn bộ văn bản sẽ trở thành một "câu" duy nhất.
        if not sentences:
            return [text.strip()]

        chunks: list[str] = []

        # Gom câu theo kích thước tối đa mỗi chunk.
        # Mỗi chunk là chuỗi nối các câu bằng một khoảng trắng để dễ đọc.
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # Văn bản rỗng thì trả về danh sách rỗng để tránh sinh chunk "ảo".
        if not text or not text.strip():
            return []

        # Nếu user truyền separators rỗng, ta fallback về "" để cắt cứng theo độ dài.
        # Điều này đảm bảo luôn có cách xử lý và không bị kẹt trong recursion.
        separators = self.separators if self.separators else [""]

        # Gọi hàm đệ quy lõi.
        raw_chunks = self._split(text, separators)

        # Chuẩn hóa đầu ra: loại chunk rỗng và loại khoảng trắng thừa ở hai đầu.
        return [c.strip() for c in raw_chunks if c and c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Luôn trim nhẹ để giảm trường hợp phát sinh chunk toàn khoảng trắng.
        current_text = current_text.strip()
        if not current_text:
            return []

        # Nếu đoạn hiện tại đã nhỏ hơn ngưỡng, giữ nguyên làm một chunk.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Không còn separator để thử -> cắt cứng theo chunk_size.
        # Đây là "điểm dừng an toàn" để recursion luôn kết thúc.
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size].strip()
                for i in range(0, len(current_text), self.chunk_size)
                if current_text[i : i + self.chunk_size].strip()
            ]

        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Separator rỗng nghĩa là yêu cầu cắt cứng ngay (ký tự-level fallback).
        if sep == "":
            return [
                current_text[i : i + self.chunk_size].strip()
                for i in range(0, len(current_text), self.chunk_size)
                if current_text[i : i + self.chunk_size].strip()
            ]

        # Nếu đoạn text không chứa separator hiện tại, thử separator thấp hơn.
        if sep not in current_text:
            return self._split(current_text, next_separators)

        # Tách theo separator hiện tại.
        # Để giữ ngữ nghĩa gần văn bản gốc, ta thêm lại separator vào cuối mỗi phần
        # (trừ phần cuối cùng).
        parts = current_text.split(sep)
        pieces: list[str] = []
        last_index = len(parts) - 1
        for idx, part in enumerate(parts):
            if idx < last_index:
                pieces.append(part + sep)
            else:
                pieces.append(part)

        chunks: list[str] = []
        buffer = ""

        # Gom các mảnh nhỏ thành chunk lớn nhất có thể nhưng không vượt chunk_size.
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            # Nếu chính mảnh này đã vượt chunk_size, cần đệ quy sâu hơn bằng separator thấp hơn.
            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer.strip())
                    buffer = ""
                chunks.extend(self._split(piece, next_separators))
                continue

            # Trường hợp còn chỗ: nối vào buffer hiện tại.
            if not buffer:
                buffer = piece
                continue

            candidate = f"{buffer} {piece}".strip()
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                # Hết chỗ: chốt buffer cũ, mở buffer mới bằng piece hiện tại.
                chunks.append(buffer.strip())
                buffer = piece

        if buffer:
            chunks.append(buffer.strip())

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Tích vô hướng (dot product) đo mức "cùng hướng" theo từng chiều.
    # Nếu hai vector càng cùng chiều, dot product càng lớn.
    dot_product = _dot(vec_a, vec_b)

    # Độ lớn (norm) của từng vector:
    # ||v|| = sqrt(sum(v_i^2))
    # Dùng math.sqrt để tính căn bậc hai.
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    # Guard quan trọng:
    # Nếu một trong hai vector có độ lớn bằng 0 (vector toàn số 0),
    # mẫu số sẽ bằng 0 -> không thể chia.
    # Theo yêu cầu bài, trả về 0.0 trong trường hợp này.
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Công thức cosine similarity:
    # cos(theta) = dot(a, b) / (||a|| * ||b||)
    # Kết quả thường nằm trong [-1, 1]:
    # 1   -> cùng hướng hoàn toàn
    # 0   -> gần vuông góc (ít liên quan)
    # -1  -> ngược hướng hoàn toàn
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Helper nhỏ để tính thống kê cho một danh sách chunk.
        # Trả về đúng cấu trúc mà phần test kỳ vọng.
        def _build_stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count > 0 else 0.0
            return {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }

        # Khởi tạo 3 chiến lược chunking built-in.
        # fixed_size: cắt theo độ dài cố định, có overlap nhẹ.
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=min(50, max(0, chunk_size // 10)))

        # by_sentences: gom theo số câu/cụm câu.
        # Heuristic: chunk_size nhỏ -> gom ít câu hơn để chunk không quá dài.
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=2 if chunk_size <= 200 else 3)

        # recursive: tách theo mức ưu tiên separator, fallback về cắt cứng.
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        # Chạy lần lượt từng strategy trên cùng một văn bản đầu vào.
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        # Trả về dict tổng hợp để dễ so sánh:
        # - count: số lượng chunk tạo ra
        # - avg_length: độ dài trung bình của chunk
        # - chunks: danh sách nội dung chunk thực tế
        return {
            "fixed_size": _build_stats(fixed_chunks),
            "by_sentences": _build_stats(sentence_chunks),
            "recursive": _build_stats(recursive_chunks),
        }

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Lưu tham chiếu tới vector store để truy xuất context liên quan câu hỏi.
        self.store = store

        # Lưu hàm LLM (dependency injection) để dễ mock trong test
        # và linh hoạt thay model trong thực tế.
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Câu hỏi rỗng hoặc chỉ có khoảng trắng -> trả thông báo ngắn,
        # tránh gọi retrieval/LLM không cần thiết.
        if not question or not question.strip():
            return "Câu hỏi trống. Vui lòng nhập câu hỏi cụ thể."

        # Bước 1: Retrieve top-k đoạn liên quan nhất từ knowledge base.
        # Mỗi phần tử kỳ vọng có các key như: content, score, metadata.
        retrieved = self.store.search(query=question, top_k=top_k)

        # Bước 2: Tạo phần context từ kết quả retrieve để đưa vào prompt.
        # Gắn số thứ tự [1], [2], ... giúp LLM dễ tham chiếu nguồn trong context.
        context_parts: list[str] = []
        for idx, item in enumerate(retrieved, start=1):
            content = (item.get("content") or "").strip()
            if content:
                context_parts.append(f"[{idx}] {content}")

        # Nếu không có context nào truy xuất được, vẫn gửi prompt fallback
        # để model trả lời rõ ràng rằng thiếu dữ liệu.
        context_text = "\n".join(context_parts) if context_parts else "(Không tìm thấy ngữ cảnh phù hợp trong kho tri thức.)"

        # Bước 3: Build prompt theo phong cách RAG.
        # Yêu cầu model ưu tiên bám sát context để giảm hallucination.
        prompt = (
            "Bạn là trợ lý trả lời dựa trên ngữ cảnh truy xuất từ kho tri thức.\n"
            "Hãy trả lời ngắn gọn, chính xác và chỉ dựa trên phần CONTEXT bên dưới.\n"
            "Nếu CONTEXT không đủ thông tin, hãy nói rõ là chưa đủ dữ liệu.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {question.strip()}\n"
            "ANSWER:"
        )

        # Bước 4: Gọi LLM để sinh câu trả lời cuối cùng.
        answer_text = self.llm_fn(prompt)

        # Chuẩn hóa đầu ra: đảm bảo luôn trả string không rỗng nếu có thể.
        if answer_text is None:
            return "Xin lỗi, tôi chưa thể tạo câu trả lời lúc này."

        final_text = str(answer_text).strip()
        return final_text or "Xin lỗi, tôi chưa thể tạo câu trả lời lúc này."

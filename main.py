from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/co-quan-quan-ly-se-giam-sat-gia-dich-vu-starlink-tai-viet-nam.txt",
    "data/cuoc-dau-cua-hai-cong-cu-ai-tai-cong-so-trung-quoc.txt",
    "data/doi_moi_sang_tao_khoi_nghiep.txt",
    "data/Nền tảng số hỗ trợ tìm nhà thầu xây dựng.txt",
    "data/phi-hanh-doan-artemis-ii-vuot-nua-duong-ve-trai-dat.txt",
    "data/Phóng thành công vệ tinh tư nhân Make in Vietnam.txt",
    "data/vn-thi-diem-doanh-nghiep-mot-nguoi.txt",
    "data/vu_tru_co_my.txt",
]


def _build_chunker(strategy: str):
    """Create a chunker instance from strategy name."""
    normalized = (strategy or "none").strip().lower()
    if normalized == "fixed":
        return FixedSizeChunker(chunk_size=700, overlap=120)
    if normalized == "sentence":
        return SentenceChunker(max_sentences_per_chunk=5)
    if normalized == "recursive":
        return RecursiveChunker(chunk_size=700)
    return None


def load_documents_from_files(file_paths: list[str], chunking_strategy: str = "none") -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    chunker = _build_chunker(chunking_strategy)

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")

        # Nếu không chọn chunking, mỗi file là một document duy nhất.
        if chunker is None:
            documents.append(
                Document(
                    id=path.stem,
                    content=content,
                    metadata={"source": str(path), "extension": path.suffix.lower(), "doc_id": path.stem},
                )
            )
            continue

        # Nếu có chunking strategy, tách file thành nhiều chunk để index chi tiết hơn.
        chunks = chunker.chunk(content)
        for idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            documents.append(
                Document(
                    id=f"{path.stem}__{idx}",
                    content=chunk_text,
                    metadata={
                        "source": str(path),
                        "extension": path.suffix.lower(),
                        "doc_id": path.stem,
                        "chunk_index": idx,
                        "chunking_strategy": chunking_strategy,
                    },
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def _context_fallback_llm(prompt: str) -> str:
    """
    Fallback không dùng API bên ngoài.

    Hàm này trích đoạn CONTEXT và trả lời ngắn gọn dựa trên nội dung đã retrieve,
    giúp tránh output kiểu demo preview khi chưa cấu hình LLM thật.
    """
    context_marker = "CONTEXT:\n"
    question_marker = "\n\nQUESTION:"

    start = prompt.find(context_marker)
    end = prompt.find(question_marker)
    if start == -1 or end == -1 or end <= start:
        return "Không đủ dữ liệu để tạo câu trả lời từ context."

    context_text = prompt[start + len(context_marker) : end].strip()
    if not context_text:
        return "Không tìm thấy ngữ cảnh phù hợp trong kho tri thức."

    # Lấy 1-2 dòng đầu của context làm câu trả lời dạng extractive.
    lines = [line.strip() for line in context_text.splitlines() if line.strip()]
    top_lines = lines[:2]
    return " ".join(top_lines)


def build_llm_from_env() -> tuple[callable, str]:
    """
    Ưu tiên LLM thật (OpenAI) nếu có OPENAI_API_KEY.

    Trả về:
    - llm_fn: hàm nhận prompt -> answer string
    - backend_name: tên backend để in ra màn hình demo
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    if not api_key:
        return _context_fallback_llm, "context fallback (khong co OPENAI_API_KEY)"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        def _openai_llm(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            content = response.choices[0].message.content
            return (content or "").strip() or "LLM khong tra ve noi dung."

        return _openai_llm, f"openai chat ({model_name})"
    except Exception:
        # Nếu OpenAI client lỗi (thiếu package hoặc lỗi runtime), fallback an toàn.
        return _context_fallback_llm, "context fallback (openai init that bai)"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."
    chunking_strategy = os.getenv("CHUNKING_STRATEGY", "none").strip().lower() or "none"

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print(f"Chunking strategy: {chunking_strategy}")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files, chunking_strategy=chunking_strategy)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    llm_fn, llm_backend = build_llm_from_env()
    print(f"LLM backend: {llm_backend}")
    if "fallback" in llm_backend:
        print("Goi y: set OPENAI_API_KEY trong .env neu ban muon cau tra loi tu LLM that.")

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())

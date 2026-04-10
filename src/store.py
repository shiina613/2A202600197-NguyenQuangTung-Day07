from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        # embedding_fn cho phép truyền hàm embed tùy chỉnh (đặc biệt hữu ích khi test).
        # Nếu không truyền vào, dùng _mock_embed mặc định để code luôn chạy được.
        self._embedding_fn = embedding_fn or _mock_embed

        # Tên collection logic của kho vector (dùng cho cả ChromaDB lẫn in-memory).
        self._collection_name = collection_name

        # Cờ cho biết hiện tại đang chạy backend nào:
        # - True  -> dùng ChromaDB
        # - False -> dùng danh sách in-memory tự quản lý
        self._use_chroma = False

        # Kho dữ liệu in-memory (fallback): mỗi phần tử là một record dict.
        self._store: list[dict[str, Any]] = []

        # Handle collection của ChromaDB (nếu khởi tạo thành công).
        self._collection = None

        # Con trỏ tăng dần để tạo id nội bộ ổn định khi cần (nhất là in-memory).
        self._next_index = 0

        # Giữ tham chiếu client để tái sử dụng ở các method khác.
        self._client = None

        try:
            # Import trong try để dự án không bắt buộc phải cài chromadb.
            import chromadb

            # Khởi tạo client mặc định (ephemeral/in-process) cho bài lab.
            client = chromadb.Client()

            # LUU Y QUAN TRONG CHO TEST:
            # get_or_create_collection sẽ "tái sử dụng" collection cũ nếu đã tồn tại.
            # Điều này có thể làm rò dữ liệu giữa các test cùng collection_name (vd: "test"),
            # khiến test "initial size is zero" bị fail do count > 0 ngay sau khi khởi tạo.
            #
            # Vì mục tiêu của bài lab là mỗi EmbeddingStore khởi tạo mới phải bắt đầu rỗng,
            # ta chủ động xóa collection cũ (nếu có) rồi tạo lại collection sạch.
            try:
                client.delete_collection(name=self._collection_name)
            except Exception:
                # Nếu collection chưa tồn tại thì bỏ qua lỗi và tiếp tục tạo mới.
                pass

            # Tạo collection mới hoàn toàn sau khi đã dọn trạng thái cũ.
            collection = client.get_or_create_collection(name=self._collection_name)

            # Chỉ bật cờ Chroma khi mọi bước trên đều thành công.
            self._client = client
            self._collection = collection
            self._use_chroma = True
        except Exception:
            # Fallback an toàn: nếu thiếu package hoặc lỗi runtime,
            # store vẫn hoạt động bằng in-memory để không làm hỏng luồng bài tập.
            self._use_chroma = False
            self._client = None
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        raise NotImplementedError("Implement EmbeddingStore._make_record")

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        raise NotImplementedError("Implement EmbeddingStore._search_records")

    def _normalize_query_field(self, raw: Any) -> list[Any]:
        """
        Chuyển dữ liệu trả về từ ChromaDB về dạng list 1 chiều ổn định.

        Lý do cần hàm này:
        - Một số phiên bản/chế độ của Chroma trả dữ liệu dạng lồng: [[...]]
        - Một số trường hợp khác có thể trả luôn dạng [...]
        - Có thể trả None nếu không có kết quả

        Hàm này giúp code phía search không phụ thuộc chặt vào shape cụ thể,
        tránh lỗi index và đảm bảo luôn build được dict kết quả có key "content".
        """
        if raw is None:
            return []

        if isinstance(raw, list):
            if not raw:
                return []

            first = raw[0]

            # Trường hợp phổ biến với Chroma cho query 1 lần:
            # - ids/documents/metadatas: [[...]]
            # - embeddings: [ndarray(shape=(k, d))]
            # Ta luôn bóc lớp "query dimension" đầu tiên để về list ứng viên 1 chiều.
            if isinstance(first, list):
                return first

            # Nếu là array-like (vd numpy.ndarray), chuyển về list Python.
            # Ví dụ: [ndarray(k, d)] -> [[...], [...], ...]
            to_list = getattr(first, "tolist", None)
            if callable(to_list):
                converted = to_list()
                if isinstance(converted, list):
                    return converted

            # Fallback: giữ nguyên cấu trúc list hiện tại.
            return raw

        return []

    def _to_vector_list(self, value: Any) -> list[float]:
        """
        Chuẩn hóa embedding về list[float] an toàn trước khi tính dot product.

        Vì sao cần hàm này:
        - Ở vài backend, embedding có thể là list Python.
        - Ở backend khác (hoặc phiên bản khác), embedding có thể là numpy array.
        - Dùng `if emb` trực tiếp với numpy array sẽ nổ lỗi:
          "truth value of an array with more than one element is ambiguous".

        Chiến lược xử lý:
        - None -> []
        - list/tuple -> ép từng phần tử sang float
        - object có `.tolist()` (ví dụ numpy array) -> đổi sang list rồi ép float
        - Trường hợp bất thường -> trả [] để tránh crash pipeline search
        """
        if value is None:
            return []

        def _normalize_sequence(seq: list[Any] | tuple[Any, ...]) -> list[float]:
            """
            Chuẩn hóa sequence về vector 1 chiều.

            Một số backend có thể trả embedding dạng lồng như [[...]] hoặc (1, d).
            Hàm này sẽ bóc lớp lồng phổ biến để thu được danh sách số thực 1 chiều.
            """
            current: Any = seq

            # Bóc lớp lồng singleton: [[v1, v2, ...]] -> [v1, v2, ...]
            while isinstance(current, (list, tuple)) and len(current) == 1 and isinstance(current[0], (list, tuple)):
                current = current[0]

            # Nếu vẫn còn lồng nhiều hàng (vd: [[...], [...]]),
            # lấy hàng đầu tiên để có vector hợp lệ cho phép tính dot product.
            if isinstance(current, (list, tuple)) and len(current) > 0 and isinstance(current[0], (list, tuple)):
                current = current[0]

            if not isinstance(current, (list, tuple)):
                return []

            return [float(x) for x in current]

        if isinstance(value, (list, tuple)):
            return _normalize_sequence(value)

        # Nhiều kiểu array-like (vd: numpy.ndarray) có method tolist().
        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            converted = to_list()
            if isinstance(converted, (list, tuple)):
                return _normalize_sequence(converted)

        return []

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # Không có tài liệu đầu vào thì không cần làm gì.
        if not docs:
            return

        # Chuẩn bị batch dữ liệu dùng cho cả 2 backend.
        # Lưu ý: id đưa vào backend nên là id nội bộ tăng dần để tránh xung đột
        # khi người dùng thêm lại document có cùng doc.id nhiều lần.
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []

        for doc in docs:
            # Sinh id nội bộ duy nhất cho từng bản ghi.
            internal_id = f"{self._collection_name}_{self._next_index}"
            self._next_index += 1

            # Tạo embedding từ nội dung văn bản.
            vector = self._embedding_fn(doc.content)

            # Metadata gốc của document (nếu có).
            # Sau đó bổ sung doc_id để các hàm filter/delete về sau xử lý thuận tiện.
            metadata = dict(doc.metadata or {})
            metadata.setdefault("doc_id", doc.id)

            ids.append(internal_id)
            documents.append(doc.content)
            metadatas.append(metadata)
            embeddings.append(vector)

        # Nhánh 1: dùng ChromaDB khi đã khởi tạo thành công ở __init__.
        if self._use_chroma and self._collection is not None:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return

        # Nhánh 2: fallback in-memory.
        # Mỗi record được chuẩn hóa để các hàm search/filter phía sau có thể dùng trực tiếp.
        for idx in range(len(ids)):
            self._store.append(
                {
                    "id": ids[idx],
                    "doc_id": metadatas[idx].get("doc_id"),
                    "content": documents[idx],
                    "metadata": metadatas[idx],
                    "embedding": embeddings[idx],
                }
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # top_k <= 0 nghĩa là không yêu cầu trả kết quả.
        if top_k <= 0:
            return []

        # Chuẩn hóa query: nếu query rỗng/chỉ khoảng trắng thì không thể search có ý nghĩa.
        if not query or not query.strip():
            return []

        # Embed query 1 lần để dùng cho toàn bộ phép so sánh.
        query_vec = self._embedding_fn(query)

        # Nhánh 1: dùng ChromaDB nếu backend này đã sẵn sàng.
        if self._use_chroma and self._collection is not None:
            # Chroma mặc định trả về distance; để nhất quán với yêu cầu bài,
            # ta chủ động tính score bằng dot product giữa query và embedding đã lưu.
            # include=['embeddings'] giúp lấy vector đã lưu để tính lại score.
            raw = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                # "ids" là trường Chroma trả sẵn; không cần đưa vào include.
                include=["documents", "metadatas", "embeddings"],
            )

            # Chuẩn hóa shape dữ liệu để tương thích nhiều phiên bản Chroma.
            ids = self._normalize_query_field(raw.get("ids"))
            docs = self._normalize_query_field(raw.get("documents"))
            metas = self._normalize_query_field(raw.get("metadatas"))
            embs = self._normalize_query_field(raw.get("embeddings"))

            results: list[dict[str, Any]] = []
            for idx in range(len(ids)):
                # Tuyệt đối không dùng `if emb` với numpy array vì sẽ gây ValueError.
                # Thay vào đó luôn chuẩn hóa về list trước, rồi kiểm tra độ dài.
                emb = self._to_vector_list(embs[idx] if idx < len(embs) else None)
                score = _dot(query_vec, emb) if len(emb) > 0 else 0.0
                results.append(
                    {
                        "id": ids[idx],
                        "content": docs[idx] if idx < len(docs) else "",
                        "metadata": metas[idx] if idx < len(metas) and metas[idx] is not None else {},
                        "score": score,
                    }
                )

            # Sắp xếp giảm dần theo score để đồng nhất hành vi với in-memory.
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        # Nhánh 2: in-memory fallback.
        # Tính score = dot(query_vec, doc_embedding) cho từng record.
        scored: list[dict[str, Any]] = []
        for record in self._store:
            # In-memory cũng đi qua chuẩn hóa để hành vi nhất quán giữa các backend.
            emb = self._to_vector_list(record.get("embedding"))
            score = _dot(query_vec, emb) if len(emb) > 0 else 0.0
            scored.append(
                {
                    "id": record.get("id"),
                    "content": record.get("content", ""),
                    "metadata": record.get("metadata", {}),
                    "score": score,
                }
            )

        # Ưu tiên các chunk có độ tương đồng cao hơn.
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # Nhánh 1: nếu đang dùng ChromaDB và collection đã sẵn sàng,
        # dùng API count() để lấy số lượng bản ghi hiện có.
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                # Nếu backend gặp lỗi tạm thời, fallback sang in-memory
                # để tránh làm hỏng toàn bộ luồng xử lý.
                return len(self._store)

        # Nhánh 2: backend in-memory -> kích thước chính là số record trong list.
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # Nếu không truyền filter (hoặc filter rỗng), hành vi giống search thông thường.
        if not metadata_filter:
            return self.search(query=query, top_k=top_k)

        # top_k <= 0 không có ý nghĩa lấy kết quả.
        if top_k <= 0:
            return []

        # Query rỗng/chỉ khoảng trắng thì không thể tính độ tương đồng hợp lệ.
        if not query or not query.strip():
            return []

        # Embed query một lần để dùng cho mọi bản ghi ứng viên sau lọc.
        query_vec = self._embedding_fn(query)

        # Hàm kiểm tra một metadata có thỏa tất cả điều kiện filter hay không.
        # Quy ước filter dạng exact-match theo từng cặp key/value.
        def _matches(meta: dict[str, Any]) -> bool:
            for key, value in metadata_filter.items():
                if meta.get(key) != value:
                    return False
            return True

        # Nhánh 1: ChromaDB.
        if self._use_chroma and self._collection is not None:
            # Tận dụng where để pre-filter ngay từ DB.
            # Sau đó tự tính lại score bằng dot product để thống nhất với search().
            raw = self._collection.query(
                query_embeddings=[query_vec],
                where=metadata_filter,
                n_results=top_k,
                # "ids" trả về mặc định; include chỉ cần các trường phụ trợ.
                include=["documents", "metadatas", "embeddings"],
            )

            # Chuẩn hóa shape để tránh lỗi khi backend đổi format kết quả.
            ids = self._normalize_query_field(raw.get("ids"))
            docs = self._normalize_query_field(raw.get("documents"))
            metas = self._normalize_query_field(raw.get("metadatas"))
            embs = self._normalize_query_field(raw.get("embeddings"))

            results: list[dict[str, Any]] = []
            for idx in range(len(ids)):
                metadata = metas[idx] if idx < len(metas) and metas[idx] is not None else {}
                # Tránh lỗi truth-value mơ hồ với numpy array bằng cách chuẩn hóa trước.
                emb = self._to_vector_list(embs[idx] if idx < len(embs) else None)
                score = _dot(query_vec, emb) if len(emb) > 0 else 0.0
                results.append(
                    {
                        "id": ids[idx],
                        "content": docs[idx] if idx < len(docs) else "",
                        "metadata": metadata,
                        "score": score,
                    }
                )

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        # Nhánh 2: in-memory fallback.
        # Bước 1: lọc record theo metadata_filter.
        filtered_records = [r for r in self._store if _matches(r.get("metadata", {}))]

        # Bước 2: chỉ search trên tập đã lọc.
        scored: list[dict[str, Any]] = []
        for record in filtered_records:
            emb = self._to_vector_list(record.get("embedding"))
            score = _dot(query_vec, emb) if len(emb) > 0 else 0.0
            scored.append(
                {
                    "id": record.get("id"),
                    "content": record.get("content", ""),
                    "metadata": record.get("metadata", {}),
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # doc_id rỗng thì xem như không có mục tiêu hợp lệ để xóa.
        if not doc_id:
            return False

        # Nhánh 1: backend ChromaDB.
        if self._use_chroma and self._collection is not None:
            try:
                # Lấy trước danh sách id cần xóa theo metadata.doc_id.
                # Chỉ cần include "ids" để nhẹ và nhanh.
                found = self._collection.get(where={"doc_id": doc_id}, include=[])
                ids = found.get("ids", []) if isinstance(found, dict) else []

                # Không tìm thấy bản ghi nào khớp -> xóa thất bại theo nghĩa "không có gì để xóa".
                if not ids:
                    return False

                # Xóa toàn bộ bản ghi thuộc cùng doc_id.
                self._collection.delete(ids=ids)
                return True
            except Exception:
                # Nếu Chroma lỗi runtime, fallback sang in-memory để vẫn có cơ hội xóa.
                pass

        # Nhánh 2: backend in-memory.
        # Giữ lại các record KHÔNG thuộc doc_id cần xóa.
        before = len(self._store)
        self._store = [
            record
            for record in self._store
            if (record.get("metadata", {}) or {}).get("doc_id") != doc_id
        ]
        after = len(self._store)

        # Trả True nếu số lượng record giảm (tức là có xóa thực sự).
        return after < before

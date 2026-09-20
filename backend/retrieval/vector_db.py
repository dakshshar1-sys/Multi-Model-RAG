import os
import logging
from langchain_community.vectorstores import FAISS
from retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from models.embedding import LocalEmbeddingModel
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# FAISS needs one document to initialise; this placeholder must never be retrieved
# or indexed lexically.
PLACEHOLDER_TEXT = "Initial empty document."


class VectorDatabase:
    """Vector database for storing and retrieving document embeddings using FAISS."""
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        # Lexical side of hybrid retrieval, rebuilt lazily whenever the store changes.
        self._store_version = 0
        self._bm25: BM25Index | None = None
        self._bm25_version = -1
        self._content_to_id: dict[str, str] = {}
        self.embeddings = LocalEmbeddingModel()
        self.vector_store = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        if os.path.exists(self.index_path) and os.listdir(self.index_path):
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}...")
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                logger.info("FAISS index loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading index: {e}. Creating new index...")
                self._create_empty_index()
        else:
            logger.info(f"No existing index found at {self.index_path}. Creating new index...")
            self._create_empty_index()

    def _create_empty_index(self):
        """Create a new empty FAISS index with a placeholder document."""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            # FAISS needs at least one document to initialize
            empty_doc = Document(page_content=PLACEHOLDER_TEXT, metadata={"source": "system"})
            self.vector_store = FAISS.from_documents([empty_doc], self.embeddings)
            logger.info("New FAISS index created successfully.")
        except Exception as e:
            logger.error(f"Failed to create empty index: {e}")
            raise

    def add_documents(self, documents: list[Document]):
        """
        Embed and add documents to FAISS.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return
        
        try:
            logger.info(f"Adding {len(documents)} documents to FAISS index...")
            self.vector_store.add_documents(documents)
            self._store_version += 1
            self.save_index()
            logger.info(f"Successfully added {len(documents)} documents to FAISS index.")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def delete_by_source(self, source: str) -> int:
        """
        Remove every chunk whose metadata.source equals `source`. Re-ingesting a file
        (e.g. after fixing its extraction) must replace its old chunks, or the garbage
        keeps competing with the good text at retrieval time. Returns chunks removed.
        """
        if not self.vector_store or not source:
            return 0
        ids = [doc_id for doc_id in self.vector_store.index_to_docstore_id.values()
               if (self.vector_store.docstore.search(doc_id).metadata or {}).get("source") == source]
        if not ids:
            return 0
        self.vector_store.delete(ids)
        self._store_version += 1
        self.save_index()
        logger.info(f"Deleted {len(ids)} chunks for source '{source}'.")
        return len(ids)

    # ── hybrid retrieval ──────────────────────────────────────────────────
    def _all_chunks(self) -> list[tuple[str, Document]]:
        vs = self.vector_store
        if not vs:
            return []
        out = []
        for doc_id in vs.index_to_docstore_id.values():
            doc = vs.docstore.search(doc_id)
            if isinstance(doc, Document) and doc.page_content and doc.page_content.strip() \
                    and doc.page_content.strip() != PLACEHOLDER_TEXT:
                out.append((doc_id, doc))
        return out

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is None or self._bm25_version != self._store_version:
            chunks = self._all_chunks()
            self._bm25 = BM25Index().build([(doc_id, d.page_content) for doc_id, d in chunks])
            self._content_to_id = {d.page_content: doc_id for doc_id, d in chunks}
            self._bm25_version = self._store_version
            logger.info(f"BM25 index built over {len(self._bm25)} chunks.")
        return self._bm25

    def bm25_retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Lexical-only retrieval (for ablations)."""
        bm25 = self._ensure_bm25()
        docstore = self.vector_store.docstore
        return [docstore.search(doc_id) for doc_id, _ in bm25.search(query, top_k)]

    def hybrid_retrieve(self, query: str, top_k: int = 5, pool: int | None = None, rrf_k: int = 60) -> list[Document]:
        """
        Dense + lexical retrieval fused with reciprocal rank fusion. Each side
        contributes a candidate pool (default 2*top_k, min 20); the fused order is
        cut to top_k. Falls back to dense-only if the lexical side has nothing.
        """
        pool = pool or max(2 * top_k, 20)
        dense_docs = self.retrieve(query, top_k=pool)
        bm25 = self._ensure_bm25()
        docstore = self.vector_store.docstore
        dense_ids = [self._content_to_id.get(d.page_content) for d in dense_docs]
        dense_ids = [i for i in dense_ids if i]
        lex_ids = [doc_id for doc_id, _ in bm25.search(query, pool)]
        if not lex_ids:
            return dense_docs[:top_k]
        fused = reciprocal_rank_fusion([dense_ids, lex_ids], k=rrf_k)
        return [docstore.search(doc_id) for doc_id in fused[:top_k]]

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Retrieve top_k documents based on vector similarity.
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized.")
            return []
        
        try:
            logger.debug(f"Retrieving top {top_k} documents for query: {query[:100]}...")
            docs = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(docs)} documents from FAISS.")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
        
    def save_index(self):
        """Save the FAISS index to disk."""
        if not self.vector_store:
            logger.warning("No vector store to save.")
            return
        
        try:
            logger.debug(f"Saving FAISS index to {self.index_path}...")
            self.vector_store.save_local(self.index_path)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

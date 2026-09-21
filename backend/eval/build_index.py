"""
Build the harness's own FAISS index from eval/corpus/, using the SAME embedding
model, splitter settings and vector-store class as production, at a separate path
so evaluation never touches the live knowledge base.

    cd backend && python -m eval.build_index
"""
from __future__ import annotations

import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from ingestion.document_parser import DocumentParser
from retrieval.vector_db import VectorDatabase

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
INDEX_DIR = os.path.join(HERE, "index")

logger = logging.getLogger(__name__)


def load_corpus() -> list[Document]:
    docs = []
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.endswith((".txt", ".md")):
            continue
        with open(os.path.join(CORPUS_DIR, name), encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": name}))
    return docs


def build(force: bool = True) -> VectorDatabase:
    if force and os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    parser = DocumentParser()                       # production chunking: 1000 / 200
    chunks = parser.text_splitter.split_documents(load_corpus())
    db = VectorDatabase(index_path=INDEX_DIR)       # production embeddings + store
    db.add_documents(chunks)
    db.save_index()
    logger.info(f"Eval index: {len(chunks)} chunks from {len(load_corpus())} documents -> {INDEX_DIR}")
    return db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db = build()
    print(f"built {INDEX_DIR}")

import os
import logging
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Parses uploaded documents and splits them into overlapping text chunks
    optimized for vector similarity search and RAG retrieval quality.
    This class currently supports PDF, TXT, and Markdown files.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.last_report = None          # ExtractionReport of the most recent PDF parse
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(f"Initialized DocumentParser with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    async def parse_upload_file(self, file: UploadFile) -> list[Document]:
        """
        Parses an uploaded file into a list of chunked Langchain Documents.
        Uses RecursiveCharacterTextSplitter for optimal retrieval granularity.
        """
        # Save uploaded file temporarily to process it
        ext = os.path.splitext(file.filename)[1].lower()
        
        with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        documents = []
        self.last_report = None
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                # Score each page's text layer; OCR the pages whose layer is absent or
                # garbage (scanner-app OCR, image-only scans). See ingestion/pdf_ocr.py.
                from ingestion.pdf_ocr import extract_pages
                results, report = extract_pages(content, [d.page_content for d in pages])
                self.last_report = report
                for doc, res in zip(pages, results):
                    doc.page_content = res.text
                    doc.metadata["page"] = res.index + 1
                    doc.metadata["extraction"] = res.extraction
                documents = [d for d in pages if d.page_content.strip()]
                if report.ocr_pages:
                    logger.info(f"DocumentParser: OCR'd {report.ocr_pages}/{report.pages} pages of "
                                f"'{file.filename}' (quality {report.mean_ratio_before} -> {report.mean_ratio_after})")
            elif ext in [".txt", ".md"]:
                loader = TextLoader(temp_file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
                
            # Split raw pages into overlapping chunks for precise retrieval
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"DocumentParser: Split '{file.filename}' into {len(chunks)} chunks (from {len(documents)} raw pages)")
            return chunks
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

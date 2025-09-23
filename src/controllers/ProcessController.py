import logging
import os

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from models import ProcessingEnum

from .BaseController import BaseController
from .ProjectController import ProjectController
from typing import List

from dataclasses import dataclass
import re
@dataclass
class Document:
    page_content: str
    metadata: dict
logger = logging.getLogger("uvicorn.error")


class ProcessController(BaseController):
    def __init__(self, project_id, embedding_client=None):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id)
        self.embedding_client = embedding_client

    def get_file_extension(self, file_id: str) -> str:
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):
        extension = self.get_file_extension(file_id)
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        if extension == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        elif extension == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)

        return None

    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()
        return None

    def process_file_content(
        self,
        file_content: list,
        file_id: str,
        chunk_size: int = 100,
        overlap_size: int = 20,
    ):
        # text_splitter = SemanticChunker(
        #     embeddings=self.embedding_client,
        #     min_chunk_size=chunk_size
        #     # chunk_overlap=overlap_size,
        #     # length_function=len,
        #     # separators=["\n\n", "\n", " ", ""],
        # )
        file_content_text = [doc.page_content for doc in file_content]
        file_content_metadata = [doc.metadata for doc in file_content]
        # chunks = text_splitter.create_documents(
        #     file_content_text, metadatas=file_content_metadata
        # )

        chunks = self.process_simpler_splitter(
            texts=file_content_text,
            metadatas=file_content_metadata,
            chunk_size=chunk_size,
        )   
        return chunks

    def clean_extracted_text(self,text: str) -> str:
        """Clean text extracted from PDFs"""
        if not text:
            return text
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    def process_simpler_splitter(self, texts: List[str], metadatas: List[dict], chunk_size: int, splitter_tag: str="\n"):
        
        full_text = " ".join(texts)

        # split by splitter_tag
        lines = [ doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip()) > 1 ]

        chunks = []
        current_chunk = ""

        for line in lines:
            current_chunk += line + splitter_tag
            if len(current_chunk) >= chunk_size:
                chunks.append(Document(
                    page_content=self.clean_extracted_text(current_chunk.strip()),
                    metadata={}
                ))

                current_chunk = ""

        if len(current_chunk) >= 0:
            chunks.append(Document(
                page_content=self.clean_extracted_text(current_chunk),
                metadata={}
            ))

        return chunks

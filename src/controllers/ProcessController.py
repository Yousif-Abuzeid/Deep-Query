import logging
import os

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from models import ProcessingEnum

from .BaseController import BaseController
from .ProjectController import ProjectController

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
        text_splitter = SemanticChunker(
            embeddings=self.embedding_client,
            min_chunk_size=chunk_size
            # chunk_overlap=overlap_size,
            # length_function=len,
            # separators=["\n\n", "\n", " ", ""],
        )
        file_content_text = [doc.page_content for doc in file_content]
        file_content_metadata = [doc.metadata for doc in file_content]
        chunks = text_splitter.create_documents(
            file_content_text, metadatas=file_content_metadata
        )
        return chunks

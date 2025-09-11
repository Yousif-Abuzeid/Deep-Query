import json
from typing import List

from models.db_schemas import DataChunk, Project
from stores.llm.LLMEnums import DocumentTypeEnum

from .BaseController import BaseController


class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, embedding_client):
        super().__init__()
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()

    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))

    def index_into_vector_db(
        self,
        project: Project,
        chunks: List[DataChunk],
        do_reset: bool = False,
        chunks_ids: List[int] = None,
    ):

        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step 2: manage items
        text = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]

        vectors = [
            self.embedding_client.embed_text(
                text=txt, document_type=DocumentTypeEnum.DOCUMENT.value
            )
            for txt in text
        ]

        # step 3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step 4: Insert into vector db

        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=text,
            vectors=vectors,
            metadata=metadatas,
            record_ids=chunks_ids,
        )
        return True
    

    def search_vector_db_collection(
        self,
        project: Project,
        text: str,
        limit: int = 5,
    ):
        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step 2: embed the text
        vector = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.QUERY.value
        )

        # step 3: search in vector db
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
        )
        if not results:
            return False

        return json.loads(json.dumps(results, default=lambda x: x.__dict__))

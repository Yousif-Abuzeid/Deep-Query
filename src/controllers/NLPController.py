import json
from typing import List

from models.db_schemas import DataChunk, Project
from stores.llm.LLMEnums import DocumentTypeEnum

from .BaseController import BaseController
import logging

class NLPController(BaseController):

    def __init__(
        self, vectordb_client, generation_client, embedding_client, template_parser=None
    ):
        super().__init__()
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger("uvicorn")

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()

    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))

    async def index_into_vector_db(
        self,
        project: Project,
        chunks: List[DataChunk],
        do_reset: bool = False,
        chunks_ids: List[int] = None,
        page_size: int = 100,
    ):
        
        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step 2: manage items
        text = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]

        vectors = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.DOCUMENT.value
        )
        # step 3: create collection if not exists
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
        )

        # step 4: Insert into vector db

        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=text,
            vectors=vectors,
            metadata=metadatas,
            record_ids=chunks_ids,
            batch_size=page_size
        )
        return True

    async def search_vector_db_collection(
        self,
        project: Project,
        text: str,
        limit: int = 5,
    ):
        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)
        query_vector = None
        # step 2: embed the text
        vector = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.QUERY.value
        )
        if not vector or len(vector) == 0:
            self.logger.error("Failed to generate embedding for the input text.")
            return False
        # if isinstance(vector, list) and len(vector) > 0:
        #     query_vector = vector[0]

        # step 3: search in vector db
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
        )
        if not results:
            return False

        return results

    async def answer_rag_question(
        self,
        project: Project,
        query: str,
        limit: int = 5,
    ):

        # step 1: retrieve relevant documents
        retrieved_docs = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )
        if not retrieved_docs or len(retrieved_docs) == 0:
            self.logger.warning("No relevant documents found in vector DB.")
            answer = None
            full_prompt = None
            chat_history = []
            return answer, full_prompt, chat_history

        # step 2: construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        # documents_prompts = []

        # for idx, doc in retrieved_docs:
        #     documents_prompts.append(
        #         self.template_parser.get('rag', 'document_prompt',{
        #             "doc_num": idx + 1,
        #             "chunk_text": doc.text
        #         })
        #     )

        documents_prompt = "\n".join(
            [
                self.template_parser.get(
                    "rag",
                    "document_prompt",
                    {"doc_num": idx + 1, "chunk_text": self.generation_client.process_text(doc.text)},
                )
                for idx, doc in enumerate(retrieved_docs)
            ]
        )

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value
            )
        ]

        full_prompt = "\n\n".join(
            [
                documents_prompt,
                footer_prompt
            ]
        )

        answer = self.generation_client.generate_text(
            prompt=full_prompt, chat_history=chat_history, system_prompt=system_prompt
        )
        return answer, full_prompt, chat_history

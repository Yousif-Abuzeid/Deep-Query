import json
import logging
from typing import List

from models.db_schemas import DataChunk, Project
from stores.llm.LLMEnums import DocumentTypeEnum

from .BaseController import BaseController


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
        return await self.vectordb_client.delete_collection(
            collection_name=collection_name
        )

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
            batch_size=page_size,
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
        vector = None
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
        # step 0: Use llm to generate multiple queries + enhance with context
        prompt_expansion = self.template_parser.get(
            "rag", "query_expansion_prompt", {"query": query}
        )

        # Get recent chat history for context-aware query expansion
        history = await self.vectordb_client.get_chat_history(
            project_id=project.project_id
        )
        recent_context = ""
        if history and len(history) > 0:
            # Get last 3 exchanges for context
            recent_messages = history[-6:] if len(history) >= 6 else history
            recent_context = "\n".join(
                [f"{msg['role']}: {msg['message']}" for msg in recent_messages]
            )

        # Enhanced query expansion with context
        if recent_context:
            enhanced_prompt = (
                f"{prompt_expansion}\n\nRecent conversation context:\n{recent_context}"
            )
        else:
            enhanced_prompt = prompt_expansion

        queries = self.generation_client.generate_text(
            prompt=enhanced_prompt, chat_history=[]
        )
        list_of_queries = [q.strip() for q in queries.strip().split("\n") if q.strip()]

        # Add the original query to ensure it's always searched
        if query not in list_of_queries:
            list_of_queries.insert(0, query)

        # step 1: retrieve relevant documents with deduplication
        retrieved_docs = []
        seen_doc_ids = set()

        for q in list_of_queries:
            self.logger.info(f"Generated Query: {q}")
            docs = await self.search_vector_db_collection(
                project=project,
                text=q,
                limit=limit,
            )
            # Deduplicate documents based on their content or ID
            for doc in docs:
                doc_id = getattr(doc, "id", None) or hash(doc.text[:100])
                if doc_id not in seen_doc_ids:
                    retrieved_docs.append(doc)
                    seen_doc_ids.add(doc_id)

        # Sort by relevance score if available
        if retrieved_docs and hasattr(retrieved_docs[0], "score"):
            retrieved_docs.sort(key=lambda x: getattr(x, "score", 0), reverse=True)

        # Limit to top results after deduplication
        retrieved_docs = retrieved_docs[
            : limit * 2
        ]  # Allow more docs since we're deduplicating

        if not retrieved_docs or len(retrieved_docs) == 0:
            self.logger.warning("No relevant documents found in vector DB.")
            answer = None
            full_prompt = None
            chat_history = []
            return answer, full_prompt, chat_history

        # step 2: construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        # Get chat history
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value
            )
        ]
        history = await self.vectordb_client.get_chat_history(
            project_id=project.project_id
        )

        if history and len(history) > 0:
            for chat in history:
                chat_history.append(
                    self.generation_client.construct_prompt(
                        prompt=chat["message"], role=chat["role"]
                    )
                )

        documents_prompt = "\n".join(
            [
                self.template_parser.get(
                    "rag",
                    "document_prompt",
                    {
                        "doc_num": idx + 1,
                        "chunk_text": self.generation_client.process_text(doc.text),
                    },
                )
                for idx, doc in enumerate(retrieved_docs)
            ]
        )

        footer_prompt = self.template_parser.get(
            "rag", "footer_prompt", {"query": query}
        )

        # Add user query to chat history
        _ = await self.vectordb_client.update_chat_history(
            project_id=project.project_id,
            new_message={
                "role": self.generation_client.enums.USER.value,
                "message": query,
            },
        )

        full_prompt = "\n\n".join([documents_prompt, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt, chat_history=chat_history, system_prompt=system_prompt
        )

        # Update chat history in DB
        _ = await self.vectordb_client.update_chat_history(
            project_id=project.project_id,
            new_message={
                "role": self.generation_client.enums.ASSISTANT.value,
                "message": answer,
            },
        )
        return answer, full_prompt, history

    async def chat(self, project: Project, query: str):
        system_prompt = self.template_parser.get("rag", "system_prompt")

        # Get chat history
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value
            )
        ]
        history = await self.vectordb_client.get_chat_history(
            project_id=project.project_id
        )

        if history and len(history) > 0:
            for chat in history:
                chat_history.append(
                    self.generation_client.construct_prompt(
                        prompt=chat["message"], role=chat["role"]
                    )
                )

        answer = self.generation_client.generate_text(
            prompt=query, chat_history=chat_history, system_prompt=system_prompt
        )

        # Update chat history in DB
        _ = await self.vectordb_client.update_chat_history(
            project_id=project.project_id,
            new_message={
                "role": self.generation_client.enums.USER.value,
                "message": query,
            },
        )

        _ = await self.vectordb_client.update_chat_history(
            project_id=project.project_id,
            new_message={
                "role": self.generation_client.enums.ASSISTANT.value,
                "message": answer,
            },
        )
        return answer, history

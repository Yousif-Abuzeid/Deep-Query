import json
import logging
from typing import List,Dict,Any

from sqlalchemy.sql import text as sql_text
import sqlalchemy
from models.db_schemas import RetrievedDocument

from ..VectorDBEnums import (
    DistanceMethodEnums,
    PgVectorDistanceMethodEnums,
    PgVectorIndexTypeEnums,
    PgVectorTableSchemeEnums,
)
from ..VectorDBInterface import VectorDBInterface


class PGVectorProvider(VectorDBInterface):

    def __init__(
        self,
        db_client,
        generation_client=None,
        default_vector_size: int = 786,
        distance_method: str = PgVectorDistanceMethodEnums.COSINE.value,
        index_threshold: int = 100,
    ):
        self.db_client = db_client
        self.generation_client = generation_client
        self.default_vector_size = default_vector_size
        self.logger = logging.getLogger("uvicorn")
        self.index_threshold = index_threshold

        self.pgvector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value

        self.default_index_name = (
            lambda collection_name: f"{collection_name}_vector_idx"
        )
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = PgVectorDistanceMethodEnums.DOT.value
        else:
            self.distance_method = PgVectorDistanceMethodEnums.COSINE.value

    async def connect(self):
        async with self.db_client() as session:
            try:
                async with session.begin():
                    await session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
            except sqlalchemy.exc.IntegrityError as e:
                if "vector" in str(e) and "already exists" in str(e):
                    print("Vector extension already exists, continuing...")
                else:
                    raise e

    async def disconnect(self):
        pass

    async def is_collection_exists(self, collection_name: str) -> bool:
        record = None
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text(
                    "SELECT * FROM pg_tables WHERE tablename = :collection_name"
                )
                results = await session.execute(
                    list_tbl, {"collection_name": collection_name}
                )
                record = results.scalar_one_or_none()

                return record is not None

    async def list_all_collections(self) -> List:
        records = []
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text(
                    "SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix"
                )
                results = await session.execute(
                    list_tbl, {"prefix": {self.pgvector_table_prefix}}
                )
                records = results.scalars().all()

        return records

    async def get_collection_info(self, collection_name: str) -> dict:
        async with self.db_client() as session:
            async with session.begin():
                
                table_info_sql = sql_text('''
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes 
                    FROM pg_tables 
                    WHERE tablename = :collection_name
                ''')

                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')

                table_info = await session.execute(table_info_sql, {"collection_name": collection_name})
                record_count = await session.execute(count_sql)

                table_data = table_info.fetchone()
                if not table_data:
                    return None
                
                return {
                    "table_info": {
                        "schemaname": table_data[0],
                        "tablename": table_data[1],
                        "tableowner": table_data[2],
                        "tablespace": table_data[3],
                        "hasindexes": table_data[4],
                    },
                    "record_count": record_count.scalar_one(),
                }
            
    async def delete_collection(self, collection_name: str):

        async with self.db_client() as session:
            async with session.begin():
                if await self.is_collection_exists(collection_name):
                    self.logger.info(f"Dropping table {collection_name}")
                    drop_sql = sql_text(f"DROP TABLE IF EXISTS {collection_name}")
                    await session.execute(drop_sql)
                    await session.commit()
                    return True
        return False

    async def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        is_collection_exists = await self.is_collection_exists(collection_name)
        if not is_collection_exists:
            self.logger.info(
                f"Creating table {collection_name} with embedding size {embedding_size}"
            )
            async with self.db_client() as session:
                async with session.begin():
                    create_sql = sql_text(
                        f"CREATE TABLE {collection_name} ("
                        f"{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY, "
                        f"{PgVectorTableSchemeEnums.TEXT.value} text, "
                        f"{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), "
                        f"{PgVectorTableSchemeEnums.CHUNK_ID.value} INTEGER, "
                        f"{PgVectorTableSchemeEnums.METADATA.value} jsonb DEFAULT '{{}}', "
                        f"FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) REFERENCES chunks(chunk_id)"
                        ")"
                    )
                    await session.execute(create_sql)
                    await session.commit()
            return True

        return False

    async def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id: str = None,
    ):

        if not await self.is_collection_exists(collection_name):
            self.logger.error(
                f"Can not insert new record to non-existed collection: {collection_name}"
            )
            return False
        if not record_id:
            self.logger.error(
                f"record_id is required for PGVectorProvider {collection_name}"
            )
            return False

        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"

        try:
            async with self.db_client() as session:
                async with session.begin():
                    insert_sql = sql_text(
                        f"INSERT INTO {collection_name} "
                        f"({PgVectorTableSchemeEnums.TEXT.value}, {PgVectorTableSchemeEnums.VECTOR.value}, {PgVectorTableSchemeEnums.CHUNK_ID.value}, {PgVectorTableSchemeEnums.METADATA.value}) "
                        f"VALUES (:text, :vector, :chunk_id, :metadata)"
                    )
                    await session.execute(
                        insert_sql,
                        {
                            "text": text,
                            "vector": "[" + ",".join([str(v) for v in vector]) + "]",
                            "chunk_id": record_id,
                            "metadata": metadata_json,
                        },
                    )
                    await self.create_vector_index(
                        collection_name, index_type=PgVectorIndexTypeEnums.HNSW.value
                    )

                    await session.commit()
                    return True

        except Exception as e:
            self.logger.error(f"Error inserting record: {e}")
            return False

    async def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list = None,
        record_ids: list = None,
        batch_size: int = 50,
    ):

        if not await self.is_collection_exists(collection_name):
            self.logger.error(
                f"Can not insert new record to non-existed collection: {collection_name}"
            )
            return False

        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None or len(record_ids) != len(texts):
            self.logger.error(
                f"record_ids must be provided and match the length of texts for PGVectorProvider {collection_name}"
            )
            return False
        if not metadata or len(metadata) == 0:
            metadata = [None] * len(texts)
        try:
            async with self.db_client() as session:
                async with session.begin():
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i : i + batch_size]
                        batch_vectors = vectors[i : i + batch_size]
                        batch_metadata = metadata[i : i + batch_size]
                        batch_record_ids = record_ids[i : i + batch_size]

                        insert_sql = sql_text(
                            f"INSERT INTO {collection_name} "
                            f"({PgVectorTableSchemeEnums.TEXT.value}, {PgVectorTableSchemeEnums.VECTOR.value}, {PgVectorTableSchemeEnums.CHUNK_ID.value}, {PgVectorTableSchemeEnums.METADATA.value}) "
                            f"VALUES (:text, :vector, :chunk_id, :metadata)"
                        )
                        values = []
                        for text, vector, metadata, record_id in zip(
                            batch_texts, batch_vectors, batch_metadata, batch_record_ids
                        ):
                            metadata_json = (
                                json.dumps(metadata, ensure_ascii=False)
                                if metadata
                                else "{}"
                            )
                            values.append(
                                {
                                    "text": text,
                                    "vector": "["
                                    + ",".join([str(v) for v in vector])
                                    + "]",
                                    "chunk_id": record_id,
                                    "metadata": metadata_json,
                                }
                            )
                        await session.execute(insert_sql, values)

                    await session.commit()
                    await self.create_vector_index(
                        collection_name,
                        index_type=PgVectorIndexTypeEnums.HNSW.value,
                    )
                    return True

        except Exception as e:
            self.logger.error(f"Error inserting batch: {e}")
            return False

    async def search_by_vector(
        self, collection_name: str, query_vector: list, limit: int = 5
    ) -> List:
        if not await self.is_collection_exists(collection_name):
            self.logger.error(
                f"Can not search in non-existed collection: {collection_name}"
            )
            return []

        vector = "[" + ",".join([str(v) for v in query_vector]) + "]"
        async with self.db_client() as session:
            async with session.begin():

                search_sql = sql_text(
                    f"SELECT {PgVectorTableSchemeEnums.ID.value}, "
                    f"{PgVectorTableSchemeEnums.TEXT.value} as text, "
                    f"1-({PgVectorTableSchemeEnums.VECTOR.value} <=> :vector) as score "
                    f"FROM {collection_name} "
                    f"ORDER BY score DESC "
                    f"LIMIT {limit}"
                )

                results = await session.execute(
                    search_sql, {"vector": vector, "limit": limit}
                )
                records = results.fetchall()
                return [
                    RetrievedDocument(
                        text=record.text,
                        score=record.score,
                    )
                    for record in records
                ]

    async def is_index_exists(self, collection_name: str) -> bool:
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                check_index_sql = sql_text(
                    """
                    SELECT 1 FROM pg_indexes
                    WHERE tablename = :collection_name
                    AND indexname = :index_name
                    """
                )
                results = await session.execute(
                    check_index_sql,
                    {"index_name": index_name, "collection_name": collection_name},
                )
                record = results.scalar_one_or_none()
                return bool(record)

    async def create_vector_index(
        self, collection_name: str, index_type: str = PgVectorIndexTypeEnums.HNSW.value
    ) -> bool:
        is_index_exists = await self.is_index_exists(collection_name)
        if is_index_exists:
            self.logger.info(f"Index already exists for collection {collection_name}")
            return False
        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
                results = await session.execute(count_sql)
                record_count = results.scalar_one()
                if record_count < self.index_threshold:
                    self.logger.info(
                        f"Not enough records to create index for collection {collection_name}"
                    )
                    return False
                self.logger.info(
                    f"START Creating index for collection {collection_name}"
                )
                index_name = self.default_index_name(collection_name)
                create_index_sql = sql_text(
                    f"CREATE INDEX {index_name} ON {collection_name} "
                    f"USING {index_type} ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})"
                )
                print(create_index_sql)
                await session.execute(create_index_sql)
                await session.commit()

                self.logger.info(f"END Creating index for collection {collection_name}")
                return True

    async def reset_vector_index(
        self, collection_name: str, index_type: str = PgVectorIndexTypeEnums.HNSW.value
    ) -> bool:
        is_index_exists = await self.is_index_exists(collection_name)
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                if is_index_exists:
                    drop_sql = sql_text(f"DROP INDEX IF EXISTS {index_name}")
                    await session.execute(drop_sql)
                    await session.commit()
        return await self.create_vector_index(collection_name, index_type)

    async def get_chat_history(self, project_id: int) -> List[Dict[str,Any]]:
        chat_history = []
        async with self.db_client() as session:
            async with session.begin():
                get_sql = sql_text(
                    "SELECT chat_history FROM projects WHERE project_id = :project_id"
                )
                results = await session.execute(
                    get_sql, {"project_id": project_id}
                )
                record = results.scalar_one_or_none()
                if record:
                    chat_history = record
        return chat_history

    async def update_chat_history(self, project_id: int, new_message: Dict[str, str]) -> bool:
        
        async with self.db_client() as session:
            async with session.begin():
                
                update_sql = sql_text("""
                UPDATE projects 
                SET chat_history = COALESCE(chat_history, '[]'::jsonb) || :new_message,
                    updated_at = NOW()
                WHERE project_id = :project_id
            """)

                await session.execute(
                    update_sql, {"new_message": json.dumps(new_message), "project_id": project_id}
                )
                await session.commit()
                return True
        return False
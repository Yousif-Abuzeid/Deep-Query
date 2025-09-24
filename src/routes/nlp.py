import logging

from controllers.NLPController import NLPController
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from models import ResponseSignal
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from routes.schemas.nlp import PushRequest, SearchRequest, ChatRequest
from tqdm.auto import tqdm
logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(prefix="/api/v1/nlp", tags=["api_v1", "nlp"])


@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    chunks_model = await ChunkModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.PROJECT_NOT_FOUND.value},
        )
    nlp_controller = NLPController(
        vectordb_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    # Create collection if not exists
    collection_name = nlp_controller.create_collection_name(
        project_id=project.project_id
    )
    _ = await request.app.vector_db_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )


    has_records = True
    page_no = 1
    inserted_items_count = 0
    page_size = 100

    idx = 0

    # Setup Batching
    total_chunks_count = await chunks_model.get_total_chunks_count(project_id=project.project_id)
    pbar = tqdm(total=total_chunks_count, desc="Indexing Chunks", unit="chunk",position=0)
    while has_records:
        chunks = await chunks_model.get_poject_chunks(
            project_id=project.project_id, page_no=page_no, page_size=page_size
        )
        page_no += 1
        if not chunks:
            has_records = False
            break

        chunks_ids = [chunk.chunk_id for chunk in chunks]
        idx += len(chunks)
        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids,
            page_size=page_size
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": ResponseSignal.INSERT_INTO_VECTOR_DB_ERROR.value},
            )
        pbar.update(len(chunks))
        inserted_items_count += len(chunks)

    return JSONResponse(
        content={
            "message": ResponseSignal.INSERT_INTO_VECTOR_DB_SUCCESS.value,
            "inserted_items_count": inserted_items_count,
        },
    )


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )
    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "message": ResponseSignal.VECTOR_DB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info,
        }
    )



@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.PROJECT_NOT_FOUND.value},
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    search_results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit,
    )
    if search_results is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.VECTOR_DB_SEARCH_ERROR.value},
        )

    return JSONResponse(
        content={
            "message": ResponseSignal.VECTOR_DB_SEARCH_SUCCESS.value,
            "results": [result.dict() for result in search_results],
        }
    )


@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.PROJECT_NOT_FOUND.value},
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )
    if answer is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.RAG_ANSWER_ERROR.value},
        )

    return JSONResponse(
        content={
            "message": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history,
        }
    )

@nlp_router.post("/index/chat/{project_id}")
async def chat(request: Request, project_id: int, chat_request: ChatRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.PROJECT_NOT_FOUND.value},
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, chat_history = await nlp_controller.chat(
        project=project,
        query=chat_request.query,
    )
    if answer is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseSignal.CHAT_ANSWER_ERROR.value},
        )

    return JSONResponse(
        content={
            "message": ResponseSignal.CHAT_ANSWER_SUCCESS.value,
            "answer": answer,  
            "chat_history": chat_history,
        }
    )
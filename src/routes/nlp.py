import logging

from controllers.NLPController import NLPController
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from models import ResponseSignal
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from routes.schemas.nlp import PushRequest, SearchRequest

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(prefix="/api/v1/nlp", tags=["api_v1", "nlp"])


@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):
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
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0

    idx = 0
    while has_records:
        chunks = await chunks_model.get_chunks_by_project_id(
            project_id=project.id, page_no=page_no
        )
        page_no += 1
        if not chunks:
            has_records = False
            break

        chunks_ids = list(range(idx, idx + len(chunks)))
        idx += len(chunks)
        is_inserted = nlp_controller.index_into_vector_db(
            project=project,
            chunks=chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids,
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": ResponseSignal.INSERT_INTO_VECTOR_DB_ERROR.value},
            )
        inserted_items_count += len(chunks)

    return JSONResponse(
        content={
            "message": ResponseSignal.INSERT_INTO_VECTOR_DB_SUCCESS.value,
            "inserted_items_count": inserted_items_count,
        },
    )


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):

    prokect_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await prokect_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )
    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "message": ResponseSignal.VECTOR_DB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info,
        }
    )



@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):
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
    )

    search_results = nlp_controller.search_vector_db_collection(
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
            "results": search_results,
        }
    )

from fastapi import APIRouter,Depends,UploadFile,status
from fastapi.responses import JSONResponse
import os
from controllers import DataController,ProjectController
from helpers.config import  get_settings,Settings
from models import ResponseSignal
import aiofiles
import logging

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"]
)



@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile, app_settings: Settings = Depends(get_settings)):
    data_controller = DataController()
    is_valid, message = data_controller.validate_uploaded_file(file)
    if not is_valid:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": message})

    
    file_path = data_controller.generate_unique_filename(file.filename, project_id)

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": ResponseSignal.FILE_UPLOAD_FAILED.value})

    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": ResponseSignal.FILE_UPLOAD_SUCCESS.value})

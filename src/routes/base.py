from fastapi import APIRouter
import os
router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)

@router.get("/")
async def welcome():
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")
    return {"message": f"Welcome to the {app_name} API v{app_version}!"}
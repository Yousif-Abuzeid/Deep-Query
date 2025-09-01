from fastapi import APIRouter,Depends
import os
from helpers.config import  get_settings,Settings
router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)

@router.get("/")
async def welcome(settings: Settings = Depends(get_settings)):
    app_name = settings.APP_NAME
    app_version = settings.APP_VERSION
    return {"message": f"Welcome to the {app_name} API v{app_version}!"}
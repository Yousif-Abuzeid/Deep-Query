from fastapi import FastAPI
from routes.base import router as base_router
from routes.data import data_router

app = FastAPI()

app.include_router(base_router)
app.include_router(data_router)
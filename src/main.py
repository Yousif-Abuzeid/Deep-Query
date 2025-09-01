from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from routes.base import router as base_router


app = FastAPI()

app.include_router(base_router)

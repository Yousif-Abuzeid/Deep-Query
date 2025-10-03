from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from helpers.config import get_settings
from routes import base_router, data_router, nlp_router
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.templates import TemplateParser
from stores.vectordb import VectorDBProviderFactory
from utils.metrics import setup_metrics

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

setup_metrics(app)


# Main page route
@app.get("/")
async def main_page():
    return FileResponse("static/index.html")


@app.on_event("startup")
async def startup_span():
    settings = get_settings()
    postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    app.db_engine = create_async_engine(postgres_conn)
    app.db_client = sessionmaker(
        app.db_engine, class_=AsyncSession, expire_on_commit=False
    )
    print("Connected to the PostgreSQL database!")
    llm_provider_factory = LLMProviderFactory(config=settings)
    vector_db_provider_factory = VectorDBProviderFactory(
        config=settings, db_client=app.db_client
    )

    # Clients
    app.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)

    # Set default models
    app.generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
    app.embedding_client.set_embedding_model(
        settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE
    )

    # Vector DB Client
    app.vector_db_client = vector_db_provider_factory.create(settings.VECTOR_DB_BACKEND)
    await app.vector_db_client.connect()

    # Template Parser
    app.template_parser = TemplateParser(
        default_language=settings.DEFAULT_LANG, language=settings.PRIMARY_LANG
    )


@app.on_event("shutdown")
async def shutdown_span():
    app.db_engine.dispose()
    print("Disconnected from the PostgreSQL database!")
    await app.vector_db_client.disconnect()


# app.router.lifespan.on_startup.append(startup_span)
# app.router.lifespan.on_shutdown.append(shutdown_span)

app.include_router(base_router)
app.include_router(data_router)
app.include_router(nlp_router)

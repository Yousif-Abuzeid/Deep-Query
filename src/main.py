from fastapi import FastAPI
from routes import base_router,data_router,nlp_router
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb import VectorDBProviderFactory



app = FastAPI()
@app.on_event("startup")
async def startup_span():
    settings = get_settings()
    app.mongodb_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongodb_conn[settings.MONGODB_DATABASE]
    print("Connected to the MongoDB database!")
    llm_provider_factory = LLMProviderFactory(config=settings)
    vector_db_provider_factory = VectorDBProviderFactory(config=settings)

    # Clients
    app.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)

    # Set default models
    app.generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
    app.embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID,settings.EMBEDDING_MODEL_SIZE)

    # Vector DB Client
    app.vector_db_client = vector_db_provider_factory.create(settings.VECTOR_DB_BACKEND)
    app.vector_db_client.connect()

@app.on_event("shutdown")
async def shutdown_span():
    app.mongodb_conn.close()
    print("Disconnected from the MongoDB database!")
    app.vector_db_client.disconnect()


# app.router.lifespan.on_startup.append(startup_span)
# app.router.lifespan.on_shutdown.append(shutdown_span)

app.include_router(base_router)
app.include_router(data_router)
app.include_router(nlp_router)
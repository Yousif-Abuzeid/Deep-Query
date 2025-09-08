# mini-rag


This is a mini version of the RAG (Retrieval-Augmented Generation) architecture.

## Requirements

- python 3.11 or higher 
- fastapi==0.116.1
- uvicorn[standard]==0.35.0
- python-multipart==0.0.20
- python-dotenv==1.1.1
- pydantic-settings==2.10.1
- aiofiles==24.1.0
- langchain==0.3.27
- PyMuPDF==1.26.4
- langchain-community==0.3.29
- motor==3.7.1

## Installation

### Install the required dependencies

```bash
pip install -r requirements.txt
```

### Setup environment variables

```bash
cp .env.example .env
```

Set your environment variables in the `.env` file. Like `OPENAI_API_KEY`.

## Run Docker Compose Services

```bash
cd docker
cp .env.example .env

```

- Set your environment variables in the `.env` file. Like `MONGO_INITDB_ROOT_USERNAME` and `MONGO_INITDB_ROOT_PASSWORD`.


### Run the application

```bash
uvicorn main:app --reload
```

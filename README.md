# mini-rag


This is a mini version of the RAG (Retrieval-Augmented Generation) architecture.

## Requirements

- python 3.11 or higher 

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

#!/usr/bin/bash

set -e

echo "Running database migrations..."
cd /app/models/db_schemas/minirag/
echo "Current directory: $(pwd)"
echo "listing files:"
ls -la
alembic upgrade head
cd /app

echo "=== DEBUG ARGUMENTS ==="
echo "Number of arguments: $#"
echo "All arguments: $@"
echo "First argument: $1"
echo "========================"

if [ $# -eq 0 ]; then
    echo "No arguments provided, starting uvicorn directly"
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Starting FastAPI application with exec..."
    exec "$@"
fi
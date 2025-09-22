#!/usr/bin/bash

set -e

echo "Running database migrations..."
cd /app/models/db_schemas/minirag/
echo "Current directory: $(pwd)"
echo "listing files:"
ls -la
alembic upgrade head
cd /app


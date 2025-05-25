#!/bin/bash
# Start the API server (Flask/FastAPI)
echo "Starting API server..."
uvicorn src/api/app:app --host 0.0.0.0 --port 8000 --reload

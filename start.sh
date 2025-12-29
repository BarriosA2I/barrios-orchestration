#!/bin/bash
set -e

echo "=============================================="
echo "  BARRIOS A2I API GATEWAY - STARTING"
echo "=============================================="

# Use PORT from environment (Render sets this)
PORT=${PORT:-8080}

echo "Starting on port: $PORT"
echo "Environment: ${ENVIRONMENT:-development}"
echo "Redis URL: ${REDIS_URL:-not set}"

# Start the API Gateway with production settings
exec uvicorn api_gateway:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 2 \
    --log-level info

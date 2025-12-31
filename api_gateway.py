#!/usr/bin/env python3
"""
API Gateway for Barrios A2I Website
Exposes Cognitive Orchestrator to Nexus Brain chat interface

Author: Barrios A2I Architecture Team
Endpoints: /api/chat, /api/video, /api/intel, /ws/events
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380")
PORT = int(os.getenv("PORT", 8080))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
import structlog
from opentelemetry import trace

# Import from Phase 1 & 2
from cognitive_orchestrator import (
    CognitiveOrchestrator,
    OrchestrationRequest,
    ProcessingMode,
    METRICS_REGISTRY,
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from event_bus import TypedEventBus, EventType, StreamChunkEvent
from websocket_bridge import WebSocketBridge

logger = structlog.get_logger("api_gateway")
tracer = trace.get_tracer("api_gateway")

# Global instances
orchestrator: Optional[CognitiveOrchestrator] = None
event_bus: Optional[TypedEventBus] = None
ws_bridge: Optional[WebSocketBridge] = None


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    """Chat request from Nexus Brain UI"""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    mode: str = "auto"
    stream: bool = False
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response to Nexus Brain UI"""
    message_id: str
    response: str
    confidence: float
    model: str
    latency_ms: float
    cost_usd: float
    citations: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class VideoRequest(BaseModel):
    """Video generation request"""
    prompt: str = Field(..., min_length=10, max_length=5000)
    style: str = "commercial"
    duration: str = "30s"
    aspect_ratio: str = "16:9"


class VideoResponse(BaseModel):
    """Video generation response"""
    job_id: str
    status: str
    progress: float = 0
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class IntelRequest(BaseModel):
    """Competitor intelligence request"""
    company_name: Optional[str] = None
    domain: Optional[str] = None
    depth: str = "standard"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float
    version: str


# ============================================================
# LIFESPAN MANAGEMENT
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle"""
    global orchestrator, event_bus, ws_bridge

    logger.info("Starting API Gateway...", environment=ENVIRONMENT, redis_url=REDIS_URL)

    # Initialize Cognitive Orchestrator
    orchestrator = CognitiveOrchestrator(
        redis_url=REDIS_URL,
        enable_circuit_breaker=True,
        enable_dlq=True,
        enable_checkpointing=True,
    )
    await orchestrator.start()

    # Initialize Event Bus
    event_bus = TypedEventBus(
        redis_url=REDIS_URL,
        enable_persistence=True,
    )
    await event_bus.start()

    # Initialize WebSocket Bridge
    ws_bridge = WebSocketBridge(event_bus)
    await ws_bridge.start()

    logger.info("API Gateway started successfully")

    yield

    # Shutdown
    logger.info("Shutting down API Gateway...")
    await ws_bridge.stop()
    await event_bus.stop()
    await orchestrator.stop()
    logger.info("API Gateway stopped")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Barrios A2I API Gateway",
    description="API Gateway for Nexus Brain chat interface",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://barrios-landing.vercel.app",
        "https://barrios-api-gateway.onrender.com",
        "https://barriosa2i.com",
        "https://www.barriosa2i.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8088",
        "*",  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CHAT ENDPOINTS
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for Nexus Brain.
    Routes through Cognitive Orchestrator with full observability.
    """
    message_id = str(uuid.uuid4())

    try:
        # Map mode
        mode_map = {
            "auto": ProcessingMode.AUTO,
            "fast": ProcessingMode.SYSTEM_1_FAST,
            "deep": ProcessingMode.SYSTEM_2_DEEP,
            "hybrid": ProcessingMode.HYBRID,
        }
        mode = mode_map.get(request.mode, ProcessingMode.AUTO)

        # Execute through orchestrator
        result = await orchestrator.execute(
            OrchestrationRequest(
                query=request.message,
                mode=mode,
                user_id=request.user_id,
                session_id=request.session_id,
                timeout_seconds=30,
                require_citations=True,
                metadata=request.context or {},
            )
        )

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error or "Query execution failed"
            )

        return ChatResponse(
            message_id=message_id,
            response=result.answer or "I couldn't generate a response.",
            confidence=result.confidence,
            model=result.model_used or "unknown",
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            citations=result.citations,
            metadata={
                "complexity": result.complexity_score,
                "mode": result.mode_used.value if result.mode_used else "auto",
                "trace_id": result.trace_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time token delivery.
    Uses true Anthropic streaming API for <3s first token latency.
    Returns Server-Sent Events (SSE) stream.
    """
    message_id = str(uuid.uuid4())

    async def generate():
        try:
            # Map mode
            mode_map = {
                "auto": ProcessingMode.AUTO,
                "fast": ProcessingMode.SYSTEM_1_FAST,
                "deep": ProcessingMode.SYSTEM_2_DEEP,
                "hybrid": ProcessingMode.HYBRID,
            }
            mode = mode_map.get(request.mode, ProcessingMode.AUTO)

            # Create orchestration request
            orch_request = OrchestrationRequest(
                query=request.message,
                mode=mode,
                session_id=request.session_id,
            )

            # Stream tokens from orchestrator using true streaming
            async for chunk in orchestrator.generate_stream(orch_request):
                # Add message_id to completion events
                if chunk.get("type") == "complete":
                    chunk["message_id"] = message_id
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            logger.error("Streaming chat error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for faster streaming
        },
    )


# ============================================================
# VIDEO ENDPOINTS
# ============================================================

@app.post("/api/video/generate", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """Generate video through RAGNAROK"""
    job_id = str(uuid.uuid4())

    # TODO: Connect to actual RAGNAROK service
    # For now, return placeholder
    return VideoResponse(
        job_id=job_id,
        status="queued",
        progress=0,
    )


@app.get("/api/video/status/{job_id}", response_model=VideoResponse)
async def video_status(job_id: str):
    """Check video generation status"""
    # TODO: Implement actual status check
    return VideoResponse(
        job_id=job_id,
        status="processing",
        progress=50,
    )


# ============================================================
# INTELLIGENCE ENDPOINTS
# ============================================================

@app.post("/api/intel/analyze")
async def analyze_competitor(request: IntelRequest):
    """Analyze competitor through Trinity"""
    # TODO: Connect to Trinity service
    return {
        "analysis_id": str(uuid.uuid4()),
        "status": "processing",
    }


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/events")
async def websocket_events(
    websocket: WebSocket,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    WebSocket endpoint for real-time event streaming.
    Clients receive events for their session/user.
    """
    client_id = await ws_bridge.connect(
        websocket,
        session_id=session_id,
        user_id=user_id,
    )

    try:
        while True:
            # Handle incoming messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        await ws_bridge.disconnect(client_id)


# ============================================================
# HEALTH & METRICS
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    health = await orchestrator.health_check()

    return HealthResponse(
        status="healthy" if health.healthy else "unhealthy",
        components=health.components,
        uptime_seconds=health.uptime_seconds,
        version=health.version,
    )


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics (JSON format)"""
    return orchestrator.get_metrics()


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format for scraping.

    Metrics exposed:
    - nexus_routing_tier_total: Request count by complexity tier
    - nexus_routing_latency_seconds: Latency histogram by tier/model
    - nexus_routing_cost_usd: Cumulative cost by tier/model
    - nexus_complexity_score: Distribution of complexity scores
    - nexus_session_messages_total: Total messages across sessions
    """
    return Response(
        content=generate_latest(METRICS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=PORT,
        reload=ENVIRONMENT == "development",
        log_level=LOG_LEVEL,
    )

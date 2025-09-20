"""Main FastAPI application for streaming-whisper."""

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from src.api.endpoints import (
    get_index,
    websocket_endpoint,
    get_transcriptions,
    get_client_list,
    delete_client_transcriptions,
    get_connection_stats
)
from src.utils.logging import (
    configure_application_logging,
    get_application_logger
)

# Configure structured logging
configure_application_logging()
logger = get_application_logger('api_main')

# Create FastAPI application
app = FastAPI(
    title="Streaming Whisper API",
    description="Real-time audio transcription using OpenAI Whisper with WebSocket support",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Basic routes
@app.get("/")
async def root():
    """Serve the main HTML page."""
    return await get_index()


@app.websocket("/ws/{client_id}")
async def websocket_route(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time transcription."""
    await websocket_endpoint(websocket, client_id)


# API routes for transcription management
@app.get("/api/transcriptions/{client_id}")
async def api_get_transcriptions(client_id: str):
    """Get all transcriptions for a specific client."""
    return await get_transcriptions(client_id)


@app.get("/api/clients")
async def api_get_clients():
    """Get list of all clients with transcription data."""
    return await get_client_list()


@app.delete("/api/transcriptions/{client_id}")
async def api_delete_transcriptions(client_id: str):
    """Delete all transcriptions for a specific client."""
    return await delete_client_transcriptions(client_id)


@app.get("/api/stats")
async def api_get_stats():
    """Get current connection and storage statistics."""
    connection_stats = await get_connection_stats()
    from src.services.transcription_service import TranscriptionStorageService
    storage_service = TranscriptionStorageService()
    storage_stats = storage_service.get_storage_stats()
    
    return {
        "connections": connection_stats,
        "storage": storage_stats
    }


# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Streaming Whisper API starting up...")
    logger.info("WebSocket endpoint available at: /ws/{client_id}")
    logger.info("Web interface available at: /")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Streaming Whisper API shutting down...")


if __name__ == "__main__":
    import uvicorn
    from src.config import server_settings
    
    logger.info(f"Starting server on {server_settings.host}:{server_settings.port}")
    uvicorn.run(
        "src.api.main:app",
        host=server_settings.host,
        port=server_settings.port,
        reload=True
    )
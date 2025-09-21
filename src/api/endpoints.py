"""API endpoints for streaming-whisper."""

import numpy as np
from typing import Dict, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.core.transcription import MicWhisper
from src.services.websocket_manager import ConnectionManager, TranscriptionWebSocketManager
from src.services.transcription_service import TranscriptionStorageService
from src.services.agent_service import AgentService
from src.utils.logging import (
    get_application_logger,
    WebSocketError,
    TranscriptionError,
    log_exception
)

logger = get_application_logger('api_endpoints')

# Initialize services
connection_manager = ConnectionManager()
transcription_ws_manager = TranscriptionWebSocketManager(connection_manager)
storage_service = TranscriptionStorageService()


async def get_index():
    """Serve the main HTML page."""
    return FileResponse("src/web/static/index.html")


async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """
    Handle WebSocket connections for real-time transcription.
    
    This endpoint manages the full lifecycle of a WebSocket connection,
    including transcription setup, audio processing, and cleanup.
    """
    # Connect the client
    client_id = await transcription_ws_manager.handle_client_connection(websocket, client_id)
    client_logger = get_application_logger('api_endpoints', client_id=client_id)
    
    # Set up transcription callback for this client
    async def send_transcription(text: str, is_final: bool):
        """Callback function to send transcription results to the client."""
        try:
            # Save final transcriptions to storage
            if is_final:
                await storage_service.save_final_transcription(client_id, text)
                
            
            # Send to client via WebSocket
            await transcription_ws_manager.send_transcription_result(client_id, text, is_final)
            
        except WebSocketError:
            # WebSocket errors are already logged by the manager
            pass
        except Exception as e:
            log_exception(client_logger, e, component="transcription_callback", client_id=client_id)
    
    # Create and configure transcriber for this client
    client_transcriber = MicWhisper()
    client_transcriber.transcription_callback = send_transcription
    
    # Store in the manager
    transcription_ws_manager.set_transcription_callback(client_id, send_transcription)
    transcription_ws_manager.set_client_transcriber(client_id, client_transcriber)
    
    try:
        # Main message handling loop
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            
            # Handle text messages
            if message.get("text") is not None:
                data = message["text"]
                client_logger.debug(f"Received text message: {data}")
                await connection_manager.broadcast(f"Client #{client_id} says: {data}")
            
            # Handle binary audio data
            elif message.get("bytes") is not None:
                audio_bytes = message["bytes"]
                size_kb = len(audio_bytes) / 1024
                client_logger.debug(f"Received {size_kb:.1f}KB of audio data")

                try:
                    # Convert raw Float32 PCM bytes to numpy array
                    # The client sends Float32Array as ArrayBuffer
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Ensure shape (frames, channels) - add channel dimension for mono
                    audio_data = audio_data.reshape(-1, 1)
                    
                    # Data is already in correct format:
                    # - 16kHz sample rate (from client)
                    # - Float32 [-1, 1] range
                    # - Shape (frames, channels)

                    client_transcriber.on_input(audio_data)
                    # Transcription results will be sent via the callback

                except (ValueError, TypeError) as e:
                    log_exception(client_logger, e, component="audio_processing", client_id=client_id)
                    try:
                        await connection_manager.send_personal_message(
                            f"[server] Invalid audio data format: {str(e)}", websocket
                        )
                    except:
                        # Don't let send errors break the main loop
                        pass
                except TranscriptionError as e:
                    log_exception(client_logger, e, component="audio_processing", client_id=client_id)
                    try:
                        await connection_manager.send_personal_message(
                            f"[server] Transcription error: {str(e)}", websocket
                        )
                    except:
                        # Don't let send errors break the main loop
                        pass
                except Exception as e:
                    log_exception(client_logger, e, component="audio_processing", client_id=client_id)
                    try:
                        await connection_manager.send_personal_message(
                            f"[server] Error processing audio: {str(e)}", websocket
                        )
                    except:
                        # WebSocket might be closed, ignore error
                        pass
            
            # Ignore other message types (pings, etc.)
    
    except WebSocketDisconnect:
        client_logger.info(f"Client {client_id} disconnected normally")
    except WebSocketError as e:
        log_exception(client_logger, e, component="websocket_main", client_id=client_id)
        # WebSocket errors are already properly logged, don't re-raise
    except Exception as e:
        log_exception(client_logger, e, component="websocket_main", client_id=client_id)
        client_logger.error(f"Unexpected error in WebSocket connection for client {client_id}: {e}")
    finally:
        # Clean up when client disconnects
        await transcription_ws_manager.handle_client_disconnect(websocket)


async def get_transcriptions(client_id: str) -> Dict[str, Any]:
    """
    Get all transcriptions for a specific client.
    
    Args:
        client_id: The client ID to get transcriptions for
        
    Returns:
        Dictionary containing all transcription data for the client
    """
    try:
        return storage_service.get_client_transcriptions(client_id)
    except Exception as e:
        log_exception(logger, e, component="get_transcriptions", client_id=client_id)
        raise


async def get_client_list() -> Dict[str, Any]:
    """
    Get list of all clients with transcription data.
    
    Returns:
        Dictionary containing client list and statistics
    """
    try:
        client_ids = storage_service.list_client_files()
        stats = storage_service.get_storage_stats()
        
        return {
            "clients": client_ids,
            "total_clients": len(client_ids),
            "storage_stats": stats
        }
    except Exception as e:
        log_exception(logger, e, component="get_client_list")
        raise


async def delete_client_transcriptions(client_id: str) -> Dict[str, Any]:
    """
    Delete all transcriptions for a specific client.
    
    Args:
        client_id: The client ID to delete transcriptions for
        
    Returns:
        Dictionary indicating success/failure
    """
    try:
        deleted = storage_service.delete_client_transcriptions(client_id)
        return {
            "success": deleted,
            "client_id": client_id,
            "message": f"Transcriptions {'deleted' if deleted else 'not found'} for client {client_id}"
        }
    except Exception as e:
        log_exception(logger, e, component="delete_transcriptions", client_id=client_id)
        raise


async def get_connection_stats() -> Dict[str, Any]:
    """
    Get current WebSocket connection statistics.
    
    Returns:
        Dictionary with connection statistics
    """
    return {
        "active_connections": connection_manager.get_connection_count(),
        "connected_clients": connection_manager.get_client_ids(),
        "transcription_callbacks": len(transcription_ws_manager.client_callbacks),
        "active_transcribers": len(transcription_ws_manager.client_transcribers)
    }
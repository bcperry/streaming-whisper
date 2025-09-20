from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import io
import os
import json
from datetime import datetime
from pathlib import Path
from transcription import MicWhisper
from src.config import storage_settings
from src.utils.logging import (
    get_application_logger, 
    configure_application_logging,
    error_handler,
    WebSocketError,
    TranscriptionError,
    log_exception
)

# Configure structured logging
configure_application_logging()
logger = get_application_logger('api')

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Store transcription callbacks per client
client_callbacks = {}


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Configuration for transcription storage
# Ensure transcription storage directory exists
Path(storage_settings.transcription_dir).mkdir(exist_ok=True)

def asses_and_act(client_id: int, text: str):
    """
    Function to assess the transcription text and perform actions.
    This function will call the action agent to determine if the primary agent needs to be invoked.
    Args:
        client_id: The ID of the client that generated the transcription
        text: The final transcription text
    """
    logger.info(f"Assessing transcription for client {client_id}: {text}")
    # Example action: Log the text length


async def handle_final_transcription(client_id: int, text: str):
    """
    Handle final transcription text for a client.
    Calls the Agent to perform the appropriate action based on the transcription.
    Then calls the helper method to save the transcription to a file.

    Args:
        client_id: The ID of the client that generated the transcription
        text: The final transcription text
    """
    logger.info(f"Handling final transcription for client {client_id}: {text}")
    # Call the helper method to save the transcription
    await save_to_file(client_id, text)
    asses_and_act(client_id, text)
    



async def save_to_file(client_id: int, text: str):
    """
    Handle final transcription text for a client.
    Saves the transcription to a client-specific JSON file with timestamp.
    
    Args:
        client_id: The ID of the client that generated the transcription
        text: The final transcription text
    """
    logger.info(f"Processing final transcription for client {client_id}: {text}")
    
    try:
        # Create filename based on client ID
        filename = f"{client_id}.json"
        filepath = Path(storage_settings.transcription_dir) / filename
        
        # Prepare the transcription entry with timestamp
        timestamp = datetime.now().isoformat()
        
        # Load existing data or create new structure
        data = {
            "all_text": "",
            "utterances": []
        }
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Ensure structure exists for backward compatibility
                if "all_text" not in data:
                    data["all_text"] = ""
                if "utterances" not in data:
                    data["utterances"] = []
            except json.JSONDecodeError as e:
                log_exception(logger, e, component="file_storage", client_id=client_id)
                logger.warning(f"Invalid JSON in transcription file for client {client_id}, creating new file")
                data = {
                    "all_text": "",
                    "utterances": []
                }
            except (OSError, IOError) as e:
                log_exception(logger, e, component="file_storage", client_id=client_id)
                logger.warning(f"Could not read transcription file for client {client_id}: {e}")
                data = {
                    "all_text": "",
                    "utterances": []
                }
        
        # Create new utterance entry
        new_utterance = {
            "timestamp": timestamp,
            "text": text
        }
        
        # Append to utterances
        data["utterances"].append(new_utterance)
        
        # Append to all_text with space separator (if not first utterance)
        if data["all_text"]:
            data["all_text"] += " " + text
        else:
            data["all_text"] = text
        
        # Save back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcription for client {client_id} to {filepath}")
        
    except (OSError, IOError) as e:
        log_exception(logger, e, component="file_storage", client_id=client_id)
        raise WebSocketError(f"Failed to save transcription for client {client_id}: {str(e)}") from e
    except json.JSONEncodeError as e:
        log_exception(logger, e, component="file_storage", client_id=client_id)
        raise WebSocketError(f"Failed to encode transcription data for client {client_id}: {str(e)}") from e
    
    # Future enhancement: Add cloud storage support here
    # This could be extended to also save to Azure Blob Storage, AWS S3, etc.
    # Example structure:
    # if CLOUD_STORAGE_ENABLED:
    #     await save_to_cloud_storage(client_id, text, timestamp)

@app.get("/")
async def get():
    return FileResponse("src/web/static/index.html")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    """Accepts both text (chat) and binary (audio chunks) frames.

    Binary frames are currently acknowledged with a short text summary.
    This is a placeholder where you can integrate real-time transcription
    (e.g., streaming Whisper) later.
    """
    await manager.connect(websocket)
    
    # Set up transcription callback for this client
    async def send_transcription(text: str, is_final: bool):
        prefix = "[final]" if is_final else "[interim]"
        logger.info(f"Sending transcription to client {client_id}: {prefix} {text}")
        try:
            # Call helper method for final transcriptions
            if is_final:
                await handle_final_transcription(client_id, text)
            await manager.send_personal_message(f"{prefix} {text}", websocket)
        except WebSocketError:
            # Re-raise WebSocket errors as they're already properly logged
            raise
        except TranscriptionError as e:
            log_exception(logger, e, component="transcription_callback", client_id=client_id)
            # Don't break the connection for transcription errors
            await manager.send_personal_message(f"[error] Transcription failed: {str(e)}", websocket)
        except Exception as e:
            log_exception(logger, e, component="websocket_send", client_id=client_id)
            raise WebSocketError(f"Failed to send transcription to client {client_id}: {str(e)}") from e
    
    # Store callback for this client and create a new transcriber instance
    client_callbacks[client_id] = send_transcription
    client_transcriber = MicWhisper()
    client_transcriber.transcription_callback = send_transcription
    logger.info(f"Set transcription callback for client {client_id}")
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            
            # message structure: {'type': 'websocket.receive', 'text': '...', 'bytes': b'...'}
            if message.get("text") is not None:
                data = message["text"].strip()
                if not data:
                    continue
                await manager.send_personal_message(f"You wrote: {data}", websocket)
                await manager.broadcast(f"Client #{client_id} says: {data}")
            elif message.get("bytes") is not None:
                audio_bytes = message["bytes"]
                size_kb = len(audio_bytes) / 1024

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
                    log_exception(logger, e, component="audio_processing", client_id=client_id)
                    try:
                        await manager.send_personal_message(
                            f"[server] Invalid audio data format: {str(e)}", websocket
                        )
                    except:
                        # Don't let send errors break the main loop
                        pass
                except TranscriptionError as e:
                    log_exception(logger, e, component="audio_processing", client_id=client_id)
                    try:
                        await manager.send_personal_message(
                            f"[server] Transcription error: {str(e)}", websocket
                        )
                    except:
                        # Don't let send errors break the main loop
                        pass
                except Exception as e:
                    log_exception(logger, e, component="audio_processing", client_id=client_id)
                    try:
                        await manager.send_personal_message(
                            f"[server] Error processing audio: {str(e)}", websocket
                        )
                    except:
                        # WebSocket might be closed, ignore error
                        pass
            # Ignore other message types (pings, etc.)
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except WebSocketError as e:
        log_exception(logger, e, component="websocket_main", client_id=client_id)
        # WebSocket errors are already properly logged, don't re-raise
    except Exception as e:
        log_exception(logger, e, component="websocket_main", client_id=client_id)
        logger.error(f"Unexpected error in WebSocket connection for client {client_id}: {e}")
    finally:
        # Clean up transcription callback when client disconnects
        if client_id in client_callbacks:
            del client_callbacks[client_id]
        client_transcriber.transcription_callback = None
        try:
            manager.disconnect(websocket)
        except ValueError:
            # Already removed from connections
            pass
        await manager.broadcast(f"Client #{client_id} left the chat")
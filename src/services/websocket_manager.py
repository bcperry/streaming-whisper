"""WebSocket connection management for streaming-whisper."""

from typing import List, Dict, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect
import json
from datetime import datetime
import uuid

from src.utils.logging import (
    get_application_logger,
    WebSocketError,
    log_exception
)

logger = get_application_logger('websocket_manager')


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        # Store active connections with client IDs
        self.active_connections: Dict[str, WebSocket] = {}
        # Reverse mapping from WebSocket to client ID
        self.connection_to_client: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            client_id: Optional client ID, will generate one if not provided
            
        Returns:
            The client ID for this connection
        """
        await websocket.accept()
        
        # Generate client ID if not provided
        if client_id is None:
            client_id = str(uuid.uuid4())
        
        # Store the connection
        self.active_connections[client_id] = websocket
        self.connection_to_client[websocket] = client_id
        
        logger.info(f"Client {client_id} connected")
        return client_id
    
    def disconnect(self, websocket: WebSocket) -> Optional[str]:
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to remove
            
        Returns:
            The client ID that was disconnected, or None if not found
        """
        client_id = self.connection_to_client.pop(websocket, None)
        if client_id and client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
        
        return client_id
    
    def get_client_id(self, websocket: WebSocket) -> Optional[str]:
        """Get the client ID for a WebSocket connection."""
        return self.connection_to_client.get(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message: The message to send
            websocket: The target WebSocket connection
        """
        try:
            await websocket.send_text(message)
        except Exception as e:
            client_id = self.get_client_id(websocket)
            log_exception(logger, e, component="send_message", client_id=client_id)
            raise WebSocketError(f"Failed to send message to client {client_id}: {str(e)}") from e
    
    async def send_personal_json(self, data: dict, websocket: WebSocket) -> None:
        """
        Send JSON data to a specific WebSocket connection.
        
        Args:
            data: The data to send as JSON
            websocket: The target WebSocket connection
        """
        try:
            await websocket.send_json(data)
        except Exception as e:
            client_id = self.get_client_id(websocket)
            log_exception(logger, e, component="send_json", client_id=client_id)
            raise WebSocketError(f"Failed to send JSON to client {client_id}: {str(e)}") from e
    
    async def broadcast(self, message: str) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: The message to broadcast
        """
        if not self.active_connections:
            logger.debug("No active connections for broadcast")
            return
        
        failed_connections = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                log_exception(logger, e, component="broadcast", client_id=client_id)
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for websocket in failed_connections:
            self.disconnect(websocket)
    
    async def broadcast_json(self, data: dict) -> None:
        """
        Broadcast JSON data to all connected clients.
        
        Args:
            data: The data to broadcast as JSON
        """
        if not self.active_connections:
            logger.debug("No active connections for JSON broadcast")
            return
        
        failed_connections = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(data)
            except Exception as e:
                log_exception(logger, e, component="broadcast_json", client_id=client_id)
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for websocket in failed_connections:
            self.disconnect(websocket)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_client_ids(self) -> List[str]:
        """Get a list of all connected client IDs."""
        return list(self.active_connections.keys())


class TranscriptionWebSocketManager:
    """Specialized WebSocket manager for transcription services."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        # Store transcription callbacks per client
        self.client_callbacks: Dict[str, Callable[[str, bool], None]] = {}
        # Store transcriber instances per client
        self.client_transcribers: Dict[str, object] = {}  # Will be MicWhisper instances
    
    async def handle_client_connection(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        Handle a new client connection for transcription.
        
        Args:
            websocket: The WebSocket connection
            client_id: Optional client ID from URL path
            
        Returns:
            The client ID for this connection
        """
        client_id = await self.connection_manager.connect(websocket, client_id)
        await self.connection_manager.broadcast(f"Client #{client_id} joined the chat")
        return client_id
    
    def set_transcription_callback(self, client_id: str, callback: Callable[[str, bool], None]) -> None:
        """
        Set the transcription callback for a client.
        
        Args:
            client_id: The client ID
            callback: The callback function to handle transcription results
        """
        self.client_callbacks[client_id] = callback
        logger.info(f"Set transcription callback for client {client_id}")
    
    def get_transcription_callback(self, client_id: str) -> Optional[Callable[[str, bool], None]]:
        """Get the transcription callback for a client."""
        return self.client_callbacks.get(client_id)
    
    def set_client_transcriber(self, client_id: str, transcriber: object) -> None:
        """
        Set the transcriber instance for a client.
        
        Args:
            client_id: The client ID
            transcriber: The MicWhisper transcriber instance
        """
        self.client_transcribers[client_id] = transcriber
        logger.info(f"Set transcriber for client {client_id}")
    
    def get_client_transcriber(self, client_id: str) -> Optional[object]:
        """Get the transcriber instance for a client."""
        return self.client_transcribers.get(client_id)
    
    async def send_transcription_result(self, client_id: str, text: str, is_final: bool) -> None:
        """
        Send a transcription result to a specific client.
        
        Args:
            client_id: The target client ID
            text: The transcribed text
            is_final: Whether this is a final or interim result
        """
        websocket = self.connection_manager.active_connections.get(client_id)
        if not websocket:
            logger.warning(f"No active connection for client {client_id}")
            return
        
        prefix = "[final]" if is_final else "[interim]"
        message = f"{prefix} {text}"
        
        try:
            await self.connection_manager.send_personal_message(message, websocket)
            logger.info(f"Sent transcription to client {client_id}: {prefix} {text}")
        except WebSocketError:
            # Already logged by connection manager
            pass
    
    def cleanup_client(self, client_id: str) -> None:
        """
        Clean up resources for a disconnected client.
        
        Args:
            client_id: The client ID to clean up
        """
        # Remove transcription callback
        if client_id in self.client_callbacks:
            del self.client_callbacks[client_id]
            logger.debug(f"Removed transcription callback for client {client_id}")
        
        # Clean up transcriber
        if client_id in self.client_transcribers:
            transcriber = self.client_transcribers[client_id]
            if hasattr(transcriber, 'transcription_callback'):
                transcriber.transcription_callback = None
            del self.client_transcribers[client_id]
            logger.debug(f"Cleaned up transcriber for client {client_id}")
    
    async def handle_client_disconnect(self, websocket: WebSocket) -> None:
        """
        Handle client disconnection and cleanup.
        
        Args:
            websocket: The disconnected WebSocket
        """
        client_id = self.connection_manager.disconnect(websocket)
        if client_id:
            self.cleanup_client(client_id)
            await self.connection_manager.broadcast(f"Client #{client_id} left the chat")
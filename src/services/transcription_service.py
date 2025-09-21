"""Storage services for transcription data."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.config import storage_settings
from src.utils.logging import (
    get_application_logger,
    WebSocketError,
    log_exception
)

logger = get_application_logger('storage_service')


class TranscriptionStorageService:
    """Service for storing and retrieving transcription data."""
    
    def __init__(self):
        # Ensure transcriptions directory exists
        self.transcriptions_dir = Path(storage_settings.transcription_dir)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Transcription storage initialized: {self.transcriptions_dir}")
    
    def _get_client_filepath(self, client_id: str) -> Path:
        """Get the file path for a client's transcriptions."""
        filename = f"{client_id}.json"
        return self.transcriptions_dir / filename
    
    def _load_client_data(self, client_id: str) -> Dict[str, Any]:
        """
        Load existing transcription data for a client.
        
        Args:
            client_id: The client ID
            
        Returns:
            Dictionary containing transcription data
        """
        filepath = self._get_client_filepath(client_id)
        
        if not filepath.exists():
            logger.debug(f"No existing transcription file for client {client_id}")
            return {
                "client_id": client_id,
                "created_at": datetime.now().isoformat(),
                "all_text": "",
                "utterances": []
            }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure structure exists for backward compatibility
            if "all_text" not in data:
                data["all_text"] = ""
            if "utterances" not in data:
                data["utterances"] = []
            if "client_id" not in data:
                data["client_id"] = client_id
            if "created_at" not in data:
                data["created_at"] = datetime.now().isoformat()
            
            logger.debug(f"Loaded existing transcription data for client {client_id}")
            return data
            
        except json.JSONDecodeError as e:
            log_exception(logger, e, component="file_storage", client_id=client_id)
            logger.warning(f"Invalid JSON in transcription file for client {client_id}, creating new file")
            return {
                "client_id": client_id,
                "created_at": datetime.now().isoformat(),
                "all_text": "",
                "utterances": []
            }
        except (OSError, IOError) as e:
            log_exception(logger, e, component="file_storage", client_id=client_id)
            logger.warning(f"Could not read transcription file for client {client_id}: {e}")
            return {
                "client_id": client_id,
                "created_at": datetime.now().isoformat(),
                "all_text": "",
                "utterances": []
            }
    
    def _save_client_data(self, client_id: str, data: Dict[str, Any]) -> None:
        """
        Save transcription data for a client.
        
        Args:
            client_id: The client ID
            data: The transcription data to save
        """
        filepath = self._get_client_filepath(client_id)
        
        try:
            # Update timestamp
            data["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved transcription for client {client_id} to {filepath}")
            
        except (OSError, IOError) as e:
            log_exception(logger, e, component="file_storage", client_id=client_id)
            raise WebSocketError(f"Failed to save transcription for client {client_id}: {str(e)}") from e
        except json.JSONEncodeError as e:
            log_exception(logger, e, component="file_storage", client_id=client_id)
            raise WebSocketError(f"Failed to encode transcription data for client {client_id}: {str(e)}") from e
    
    async def save_final_transcription(self, client_id: str, text: str) -> None:
        """
        Save a final transcription for a client.
        
        Args:
            client_id: The client ID
            text: The transcribed text
        """
        timestamp = datetime.now().isoformat()
        
        # Load existing data
        data = self._load_client_data(client_id)
        
        # Create new utterance entry
        new_utterance = {
            "timestamp": timestamp,
            "text": text
        }
        
        # Append to utterances
        data["utterances"].append(new_utterance)
        
        # Update cumulative text
        if data["all_text"]:
            data["all_text"] += " " + text
        else:
            data["all_text"] = text
        
        # Save back to file
        self._save_client_data(client_id, data)
    
    def get_client_transcriptions(self, client_id: str) -> Dict[str, Any]:
        """
        Get all transcriptions for a client.
        
        Args:
            client_id: The client ID
            
        Returns:
            Dictionary containing all transcription data
        """
        return self._load_client_data(client_id)
    
    def get_client_utterances(self, client_id: str) -> List[Dict[str, Any]]:
        """
        Get all utterances for a client.
        
        Args:
            client_id: The client ID
            
        Returns:
            List of utterance dictionaries
        """
        data = self._load_client_data(client_id)
        return data.get("utterances", [])
    
    def get_client_full_text(self, client_id: str) -> str:
        """
        Get the full transcribed text for a client.
        
        Args:
            client_id: The client ID
            
        Returns:
            The complete transcribed text
        """
        data = self._load_client_data(client_id)
        return data.get("all_text", "")
    
    def list_client_files(self) -> List[str]:
        """
        List all client transcription files.
        
        Returns:
            List of client IDs that have transcription files
        """
        client_ids = []
        
        for file_path in self.transcriptions_dir.glob("*.json"):
            # Extract client ID from filename (filename is the client ID)
            filename = file_path.stem  # Remove .json extension
            client_ids.append(filename)
        
        return sorted(client_ids)
    
    def delete_client_transcriptions(self, client_id: str) -> bool:
        """
        Delete all transcriptions for a client.
        
        Args:
            client_id: The client ID
            
        Returns:
            True if deleted successfully, False if file didn't exist
        """
        filepath = self._get_client_filepath(client_id)
        
        if not filepath.exists():
            logger.info(f"No transcription file to delete for client {client_id}")
            return False
        
        try:
            filepath.unlink()
            logger.info(f"Deleted transcription file for client {client_id}")
            return True
        except (OSError, IOError) as e:
            log_exception(logger, e, component="file_storage", client_id=client_id)
            raise WebSocketError(f"Failed to delete transcriptions for client {client_id}: {str(e)}") from e
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        client_files = self.list_client_files()
        total_files = len(client_files)
        
        total_size = 0
        total_utterances = 0
        
        for client_id in client_files:
            filepath = self._get_client_filepath(client_id)
            if filepath.exists():
                total_size += filepath.stat().st_size
                utterances = self.get_client_utterances(client_id)
                total_utterances += len(utterances)
        
        return {
            "total_clients": total_files,
            "total_files_size_bytes": total_size,
            "total_utterances": total_utterances,
            "storage_directory": str(self.transcriptions_dir)
        }
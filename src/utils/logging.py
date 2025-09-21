"""Structured logging configuration and utilities for streaming-whisper."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import traceback

from src.config.settings import LoggingSettings


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured information to log records."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record):
        # Add structured fields to the record
        if not hasattr(record, 'component'):
            record.component = record.name.split('.')[-1] if '.' in record.name else record.name
        
        if not hasattr(record, 'client_id'):
            record.client_id = getattr(record, 'client_id', 'N/A')
            
        # Format the message
        formatted = super().format(record)
        
        # Add exception details if present
        if record.exc_info:
            formatted += f"\nException Details: {self.formatException(record.exc_info)}"
            
        return formatted


class StreamingWhisperLogger:
    """Centralized logger configuration for streaming-whisper application."""
    
    def __init__(self, settings: LoggingSettings):
        self.settings = settings
        self._configured = False
        self._loggers = {}
    
    def configure_logging(self) -> None:
        """Configure application-wide logging."""
        if self._configured:
            return
            
        # Clear any existing handlers from root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set root logger level
        log_level = getattr(logging, self.settings.level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Create formatters
        detailed_formatter = StructuredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(component)-12s | %(client_id)-10s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        simple_formatter = StructuredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(component)-12s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(log_level)
        
        # Configure API log handler
        api_handler = self._create_file_handler(
            self.settings.api_log_file, 
            detailed_formatter
        )
        
        # Configure transcription log handler  
        transcription_handler = self._create_file_handler(
            self.settings.transcription_log_file,
            detailed_formatter
        )
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(api_handler)
        root_logger.addHandler(transcription_handler)
        
        # Configure specific loggers
        self._configure_module_loggers(api_handler, transcription_handler)
        
        self._configured = True
        
    def _create_file_handler(self, filename: str, formatter: logging.Formatter) -> logging.FileHandler:
        """Create a file handler with rotation support."""
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler to prevent log files from growing too large
        handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, self.settings.level.upper(), logging.INFO))
        
        return handler
        
    def _configure_module_loggers(self, api_handler: logging.Handler, transcription_handler: logging.Handler) -> None:
        """Configure specific module loggers with appropriate handlers."""
        # Create console handler for module loggers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(component)-12s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        console_handler.setLevel(getattr(logging, self.settings.level.upper(), logging.INFO))
        
        # API module logger - logs to API file and console
        api_logger = logging.getLogger('api')
        api_logger.handlers.clear()
        api_logger.addHandler(api_handler)
        api_logger.addHandler(console_handler)
        api_logger.propagate = False  # Don't propagate to root
        
        # Transcription module logger - logs to transcription file and console  
        transcription_logger = logging.getLogger('transcription')
        transcription_logger.handlers.clear()
        transcription_logger.addHandler(transcription_handler)
        transcription_logger.addHandler(console_handler)
        transcription_logger.propagate = False  # Don't propagate to root
        
        # Store configured loggers
        self._loggers['api'] = api_logger
        self._loggers['transcription'] = transcription_logger
    
    def get_logger(self, name: str, client_id: Optional[str] = None) -> logging.Logger:
        """Get a configured logger with optional client context."""
        if not self._configured:
            self.configure_logging()
            
        logger = logging.getLogger(name)
        
        # Add client_id to logger if provided
        if client_id:
            logger = logging.LoggerAdapter(logger, {'client_id': client_id})
            
        return logger


class LoggingContextManager:
    """Context manager for adding structured context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
            
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Exception handling utilities
class StreamingWhisperError(Exception):
    """Base exception for streaming-whisper application."""
    
    def __init__(self, message: str, component: str = "", client_id: Optional[str] = None):
        super().__init__(message)
        self.component = component
        self.client_id = client_id


class TranscriptionError(StreamingWhisperError):
    """Exception raised during transcription operations."""
    pass


class WebSocketError(StreamingWhisperError):
    """Exception raised during WebSocket operations."""
    pass


class ConfigurationError(StreamingWhisperError):
    """Exception raised during configuration loading."""
    pass


def log_exception(logger: logging.Logger, exc: Exception, 
                 component: str = "", client_id: Optional[str] = None) -> None:
    """Log an exception with structured information."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    
    # Create log record with structured data
    extra = {
        'component': component,
        'client_id': client_id or 'N/A',
        'exception_type': exc_type
    }
    
    logger.error(f"Exception in {component}: {exc_type}: {exc_msg}", extra=extra, exc_info=True)


@contextmanager
def error_handler(logger: logging.Logger, component: str = "", 
                 client_id: Optional[str] = None, reraise: bool = True):
    """Context manager for consistent error handling and logging."""
    try:
        yield
    except Exception as e:
        log_exception(logger, e, component, client_id)
        if reraise:
            raise


# Global logger instance
_logger_instance: Optional[StreamingWhisperLogger] = None


def get_application_logger(name: str, client_id: Optional[str] = None) -> logging.Logger:
    """Get the application logger instance."""
    global _logger_instance
    
    if _logger_instance is None:
        from src.config import logging_settings
        _logger_instance = StreamingWhisperLogger(logging_settings)
        
    return _logger_instance.get_logger(name, client_id)


def configure_application_logging() -> None:
    """Configure application-wide logging."""
    global _logger_instance
    
    if _logger_instance is None:
        from src.config import logging_settings
        _logger_instance = StreamingWhisperLogger(logging_settings)
        
    _logger_instance.configure_logging()
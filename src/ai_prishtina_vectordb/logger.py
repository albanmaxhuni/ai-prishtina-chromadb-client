"""
Logging functionality for AIPrishtina VectorDB.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

class AIPrishtinaLogger:
    """Logger class for AIPrishtina VectorDB."""
    
    def __init__(
        self,
        name: str = "ai_prishtina_vectordb",
        level: str = "INFO",
        log_file: Optional[Union[str, Path]] = None
    ):
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
        """
        self.name = str(name)  # Ensure name is a string
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        
        # Configure logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log file is specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

# Create default logger instance
logger = AIPrishtinaLogger() 
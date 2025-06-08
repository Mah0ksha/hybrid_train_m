import time
import logging
import coloredlogs  # For colored console output
from typing import Dict, List, Any, Tuple, Union # Added Union

class RateLimitedLogger:
    def __init__(self, logger: logging.Logger, interval: float = 5.0):
        self.logger: logging.Logger = logger
        self.interval: float = interval
        self.last_log: Dict[str, float] = {}
        self.buffer: List[Tuple[str, str]] = []
        self.buffer_size: int = 100
        self.last_flush: float = time.time()

    def flush_buffer(self):
        """Flush the buffer by logging all messages."""
        for item in self.buffer:
            log_type = item[0]
            message = item[1]
            if log_type == 'info':
                self.logger.info(message)
            elif log_type == 'error':
                self.logger.error(message)
            elif log_type == 'warning':
                self.logger.warning(message)
            elif log_type == 'debug':
                self.logger.debug(message)
        self.buffer.clear()
        self.last_flush = time.time()
    
    def info(self, message: str, key: Union[str, None] = None):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('info', message))
            self.last_log[key] = current_time
            
            # Flush buffer if it's full or enough time has passed
            if len(self.buffer) >= self.buffer_size or (current_time - self.last_flush) >= self.interval:
                self.flush_buffer()
            
    def error(self, message: str, key: Union[str, None] = None, **kwargs: Any):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('error', message))
            self.last_log[key] = current_time
            self.flush_buffer()  # Always flush errors immediately
            
    def warning(self, message: str, key: Union[str, None] = None):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('warning', message))
            self.last_log[key] = current_time
            if len(self.buffer) >= self.buffer_size or (current_time - self.last_flush) >= self.interval:
                self.flush_buffer()

    def debug(self, message: str, key: Union[str, None] = None):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('debug', message))
            self.last_log[key] = current_time
            if len(self.buffer) >= self.buffer_size or (current_time - self.last_flush) >= self.interval:
                self.flush_buffer()

    # Logger interface compatibility methods
    def log(self, level: int, message: str, *args, **kwargs):
        """Implement log method for compatibility with Logger interface."""
        if level >= logging.ERROR:
            self.error(message, **kwargs)
        elif level >= logging.WARNING:
            self.warning(message)
        elif level >= logging.INFO:
            self.info(message)
        else:
            self.debug(message)
            
    def getEffectiveLevel(self) -> int:
        """Implement getEffectiveLevel for compatibility with Logger interface."""
        return self.logger.getEffectiveLevel()

    def setLevel(self, level: int) -> None:
        """Implement setLevel for compatibility with Logger interface."""
        self.logger.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        """Implement isEnabledFor for compatibility with Logger interface."""
        return self.logger.isEnabledFor(level)

    def getChild(self, suffix: str) -> 'RateLimitedLogger':
        """Implement getChild for compatibility with Logger interface."""
        child_logger = self.logger.getChild(suffix)
        return RateLimitedLogger(child_logger, self.interval)

    def addHandler(self, handler: logging.Handler) -> None:
        """Implement addHandler for compatibility with Logger interface."""
        self.logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        """Implement removeHandler for compatibility with Logger interface."""
        self.logger.removeHandler(handler)

def setup_logger() -> RateLimitedLogger:
    """Set up and return a rate-limited logger."""
    try:
        # Silence external library logs
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

        # Get the 'main' logger
        logger = logging.getLogger('main')
        logger.handlers.clear()  # Remove any existing handlers
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Create console handler with color
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)

        # Create formatter and add it to the console handler
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
        console.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console)

        # Create file handler
        try:
            file_handler = logging.FileHandler('trading_run.log', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter) # Use the same formatter for the file
            logger.addHandler(file_handler)
            logger.info("File logger initialized at trading_run.log")
        except Exception as e_file:
            logger.error(f"Failed to initialize file logger: {e_file}")

        # Create and return rate-limited logger
        return RateLimitedLogger(logger, interval=5.0)
        
    except Exception as e:
        # Fallback to basic logging if there's an error
        logging.basicConfig(level=logging.INFO)
        _logger = logging.getLogger('main')  # Use a different variable name to avoid confusion
        _logger.error(f"Error setting up logger: {str(e)}")
        return RateLimitedLogger(_logger, interval=5.0)  # Still wrap in RateLimitedLogger for consistency

# Initialize the logger when this module is imported
logger: Union[RateLimitedLogger, logging.Logger] = setup_logger() # Global logger's type
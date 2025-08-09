import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
import json


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RAGAgentError(Exception):
    """Base exception for RAG Agent"""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "type": self.__class__.__name__,
        }


class DocumentProcessingError(RAGAgentError):
    """Error during document processing"""

    def __init__(
        self,
        message: str,
        document_path: str,
        error_code: str = "DOCUMENT_PROCESSING_ERROR",
    ):
        super().__init__(message, error_code, {"document_path": document_path})


class AgentError(RAGAgentError):
    """Error in agent execution"""

    def __init__(self, message: str, agent_name: str, error_code: str = "AGENT_ERROR"):
        super().__init__(message, error_code, {"agent_name": agent_name})


class ConfigurationError(RAGAgentError):
    """Error in configuration"""

    def __init__(
        self,
        message: str,
        config_key: str = None,
        error_code: str = "CONFIGURATION_ERROR",
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, error_code, details)


class OllamaConnectionError(RAGAgentError):
    """Error connecting to Ollama"""

    def __init__(
        self, message: str, endpoint: str, error_code: str = "OLLAMA_CONNECTION_ERROR"
    ):
        super().__init__(message, error_code, {"endpoint": endpoint})


class ValidationError(RAGAgentError):
    """Error in data validation"""

    def __init__(
        self, message: str, field_name: str = None, error_code: str = "VALIDATION_ERROR"
    ):
        details = {}
        if field_name:
            details["field_name"] = field_name
        super().__init__(message, error_code, details)


class RAGAgentFormatter(logging.Formatter):
    """Custom formatter for RAG Agent logs"""

    def format(self, record):
        # Add timestamp if not present
        if not hasattr(record, "timestamp"):
            record.timestamp = datetime.now().isoformat()

        # Add error details if present
        if hasattr(record, "error_details"):
            record.error_details = json.dumps(record.error_details, default=str)

        return super().format(record)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class RAGAgentLogger:
    """Enhanced logger for RAG Agent"""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = log_dir or Path.home() / ".ragagent" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup logging handlers"""

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)

        # File handler for all logs
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = RAGAgentFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Error file handler for errors only
        error_log_file = self.log_dir / f"{self.name}_error.log"
        error_handler = logging.FileHandler(error_log_file, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context"""

        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }

        self.logger.error(
            f"Error occurred: {error}", extra={"error_details": error_details}
        )

    def log_agent_execution(
        self,
        agent_name: str,
        action: str,
        duration: float,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log agent execution"""

        log_data = {
            "agent": agent_name,
            "action": action,
            "duration": duration,
            "success": success,
            "details": details or {},
        }

        if success:
            self.logger.info(
                f"Agent {agent_name} {action} completed in {duration:.2f}s"
            )
        else:
            self.logger.error(f"Agent {agent_name} {action} failed in {duration:.2f}s")

        self.logger.debug(f"Agent execution details: {json.dumps(log_data, indent=2)}")

    def log_document_processing(
        self,
        document_path: str,
        action: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log document processing"""

        log_data = {
            "document_path": document_path,
            "action": action,
            "success": success,
            "details": details or {},
        }

        if success:
            self.logger.info(f"Document {document_path} {action} completed")
        else:
            self.logger.error(f"Document {document_path} {action} failed")

        self.logger.debug(
            f"Document processing details: {json.dumps(log_data, indent=2)}"
        )

    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_data: Optional[Dict[str, Any]] = None,
    ):
        """Log API request"""

        log_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration": duration,
            "request_data": request_data or {},
        }

        if status_code < 400:
            self.logger.info(
                f"API {method} {endpoint} - {status_code} in {duration:.2f}s"
            )
        else:
            self.logger.error(
                f"API {method} {endpoint} - {status_code} in {duration:.2f}s"
            )

        self.logger.debug(f"API request details: {json.dumps(log_data, indent=2)}")

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Log performance metrics"""

        log_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
        }

        self.logger.info(f"Performance: {metric_name} = {value} {unit}")
        self.logger.debug(f"Performance details: {json.dumps(log_data, indent=2)}")

    def debug(self, message: str, **kwargs):
        """Debug log"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Info log"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Warning log"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Error log"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Critical log"""
        self.logger.critical(message, **kwargs)


def get_logger(name: str, log_dir: Optional[Path] = None) -> RAGAgentLogger:
    """Get a logger instance"""
    return RAGAgentLogger(name, log_dir)


class LogContext:
    """Context manager for logging operations"""

    def __init__(
        self,
        logger: RAGAgentLogger,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger
        self.operation = operation
        self.context = context or {}
        self.start_time = None
        self.success = False

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}")
        self.logger.debug(f"Operation context: {self.context}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.success = True
            self.logger.info(f"Completed {self.operation} in {duration:.2f}s")
        else:
            self.success = False
            self.logger.error(f"Failed {self.operation} in {duration:.2f}s")
            self.logger.log_error(exc_val, self.context)

        # Don't suppress exceptions
        return False


class PerformanceTracker:
    """Track performance metrics"""

    def __init__(self, logger: RAGAgentLogger):
        self.logger = logger
        self.metrics = {}

    def start_timer(self, name: str):
        """Start a timer"""
        self.metrics[name] = {
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None,
        }

    def end_timer(self, name: str):
        """End a timer and log the duration"""
        if name in self.metrics:
            self.metrics[name]["end_time"] = datetime.now()
            self.metrics[name]["duration"] = (
                self.metrics[name]["end_time"] - self.metrics[name]["start_time"]
            ).total_seconds()

            duration = self.metrics[name]["duration"]
            self.logger.log_performance_metric(name, duration, "seconds")

    def get_duration(self, name: str) -> Optional[float]:
        """Get the duration of a timer"""
        if name in self.metrics and self.metrics[name]["duration"]:
            return self.metrics[name]["duration"]
        return None

    def log_all_metrics(self):
        """Log all metrics"""
        for name, metric in self.metrics.items():
            if metric["duration"]:
                self.logger.log_performance_metric(name, metric["duration"], "seconds")

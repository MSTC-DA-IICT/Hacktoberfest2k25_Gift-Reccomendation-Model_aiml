"""
Centralized logging configuration.

This module provides logging utilities and configuration for the entire application.
"""

import os
import logging
import logging.handlers
from typing import Optional
import yaml


def setup_logger(name: str, config_path: str = "config/config.yaml",
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with configuration from YAML file.

    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    config_path : str, default="config/config.yaml"
        Path to configuration file
    log_file : Optional[str]
        Override log file path

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("This is a test message")
    """
    try:
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            logging_config = config.get('logging', {})
        else:
            # Default configuration if file not found
            logging_config = {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/app.log'
            }

        # Create logger
        logger = logging.getLogger(name)

        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Set level
        level_str = logging_config.get('level', 'INFO')
        level = getattr(logging, level_str.upper(), logging.INFO)
        logger.setLevel(level)

        # Create formatter
        format_str = logging_config.get('format',
                                       '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(format_str)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file_path = log_file or logging_config.get('file', 'logs/app.log')
        if log_file_path:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

        return logger

    except Exception as e:
        # Fallback to basic logging if setup fails
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        logger.error(f"Error setting up logger: {e}")
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return setup_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Examples
    --------
    >>> class MyClass(LoggerMixin):
    ...     def do_something(self):
    ...         self.logger.info("Doing something...")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger


def log_execution_time(func):
    """
    Decorator to log function execution time.

    Examples
    --------
    >>> @log_execution_time
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise

    return wrapper


def configure_third_party_loggers(level: str = "WARNING") -> None:
    """
    Configure third-party library loggers to reduce noise.

    Parameters
    ----------
    level : str, default="WARNING"
        Log level for third-party loggers
    """
    third_party_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'matplotlib',
        'PIL',
        'cv2'
    ]

    log_level = getattr(logging, level.upper(), logging.WARNING)

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(log_level)


# Initialize third-party logger configuration
configure_third_party_loggers()
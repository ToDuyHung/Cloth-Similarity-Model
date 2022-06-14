import os
import logging
from logging.handlers import RotatingFileHandler

from config.config import Config


def setup_logging():
    config = Config()
    if not os.path.exists('logs'):
        os.mkdir('logs')
    new_format = "%(asctime)s | %(name)s | [%(levelname)s] | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=new_format,
        handlers=[
            RotatingFileHandler(config.log_file, encoding="utf8",
                                maxBytes=1024*10240, backupCount=10),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(config.perf_logger_name)
    file_logger_1 = RotatingFileHandler(config.perf_log_file, encoding="utf8", maxBytes=1024*10240, backupCount=10)
    file_logger_format_1 = logging.Formatter(new_format)
    file_logger_1.setFormatter(file_logger_format_1)
    logger.addHandler(file_logger_1)
    logger.setLevel(logging.INFO)

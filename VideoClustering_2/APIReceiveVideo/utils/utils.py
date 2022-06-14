import os
import logging
from logging.handlers import RotatingFileHandler
import uuid

from config.config import Config


def generate_id(text):
    uuid4 = uuid.uuid4()
    uuid5 = uuid.uuid5(uuid4, text)
    return uuid5


def setup_logging():
    config = Config()
    os.makedirs(config.logging_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | [%(levelname)s] | %(message)s",
        handlers=[
            RotatingFileHandler(os.path.join(config.logging_folder, "app.log"), encoding="utf8",
                                maxBytes=1024*10, backupCount=10),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
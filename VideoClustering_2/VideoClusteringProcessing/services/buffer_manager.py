import logging
from queue import Queue

from config.config import Config
from common.queue_name import QName
from objects.singleton import Singleton


class BufferManager(metaclass=Singleton):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()
        self.qs = {
            QName.MESSAGE_VIDEO_Q: Queue(maxsize=int(self.config.number_video_readers)),
            QName.PROCESS_VIDEO_Q: Queue(maxsize=int(self.config.number_of_workers / 2) + 1),
            QName.ADD_DB_Q: Queue(),
            QName.OUT_IMAGE_Q: Queue()
        }

    def get_data(self, queue_name: QName):
        if queue_name in self.qs:
            output = self.qs[queue_name].get()
            self.logger.info(f"GET ITEM FROM {queue_name}. {self.qs[queue_name].qsize()} ITEMS REMAINS")
            return output
        raise ValueError("Queue name not in Queue list")

    def put_data(self, queue_name: QName, data):
        if queue_name in self.qs:
            self.qs[queue_name].put(data)
            self.logger.info(f"PUT ITEM TO {queue_name}. {self.qs[queue_name].qsize()} ITEMS REMAINS")
        else:
            raise ValueError("Queue name not in Queue list")

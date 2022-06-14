import logging
import requests

from config.config import Config
from objects.singleton import Singleton


class BaseSingleton(metaclass=Singleton):
    pass


class BaseService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()
        self.session = None
        self.logger.info(f"CREATE {self.__class__.__name__}")

    def init_session(self, force=False):
        if self.session is None or force:
            self.session = requests.session()

    def call_back_func(self):
        self.init_session(True)

    def make_request(self, request_func, call_back_func=None):
        self.init_session(False)
        result = None
        num_retry = self.config.num_retry
        while result is None and num_retry > 0:
            try:
                result = request_func()
            except Exception as e:
                self.logger.error(f"Cannot make request. Error : {e}")
                if call_back_func is not None:
                    call_back_func()
                else:
                    self.call_back_func()
                num_retry -= 1
        return result


class BaseServiceSingleton(BaseService, BaseSingleton):
    pass

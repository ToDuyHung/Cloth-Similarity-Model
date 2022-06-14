import os

from objects.singleton import Singleton
from common.common_keys import *


class Config(metaclass=Singleton):
    manager_api = os.getenv(MANAGER_API, "http://172.29.5.91:8080/video/info")
    media_server_api = os.getenv(MEDIA_SERVER_API, "http://172.29.13.24:35007/hls/")
    _original_video_folder = os.getenv(ORIGINAL_VIDEO_FOLDER, "data/original")
    _resampled_video_folder = os.getenv(RESAMPLED_VIDEO_FOLDER, "data/resampled")
    logging_folder = os.getenv(LOGGING_FOLDER, "logs")
    original_video_time_alive = int(os.getenv(ORIGINAL_VIDEO_TIME_ALIVE, 3600))
    num_retry = int(os.getenv(NUM_RETRY, 5))

    @property
    def original_video_folder(self):
        os.makedirs(self._original_video_folder, exist_ok=True)
        return self._original_video_folder

    @property
    def resampled_video_folder(self):
        os.makedirs(self._resampled_video_folder, exist_ok=True)
        return self._resampled_video_folder

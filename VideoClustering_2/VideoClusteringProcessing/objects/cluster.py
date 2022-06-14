from typing import List

from config.config import Config
from common.common_keys import *


class Cluster:

    def __init__(self, video_id: str, list_frame: List[int] = None, key_frame: int = None, label: int = None):
        self.config = Config()
        self.video_id = video_id
        self.list_frame = list_frame
        self.key_frame = key_frame
        self.label = label
        self._image_url = None

    @property
    def image_url(self):
        return f'{self.config.file_server_get_url}video_{self.video_id}_{self.label}.jpg'

    @property
    def info(self):
        return {
            ID: self.label,
            IMAGE_URL: self.image_url,
            BEGIN: self.list_frame[0],
            END: self.list_frame[-1]
        }

    def delete(self):
        self.config = None
        self.list_frame = None
        self.key_frame = None
        self.label = None
        self._image_url = None

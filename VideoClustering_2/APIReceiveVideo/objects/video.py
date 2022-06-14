from datetime import datetime
from fastapi import UploadFile, File

from common.video_status import VideoStatus
from utils.utils import generate_id


class Video:
    def __init__(self, data: UploadFile = File('file')):
        self.data = data
        self.name, self.extension = data.filename.split(".")
        self.id = f"{generate_id(self.name)}@{self.name}"
        self.status = VideoStatus.CREATED
        self.sampling_time = None
        self.created_time = int(datetime.now().timestamp()*1000)

    @property
    def is_success(self):
        return self.status == VideoStatus.SUCCESS

    @property
    def file_name(self):
        return self.data.filename

    @property
    def file(self):
        return self.data.file

    @property
    def m3u8(self):
        return f"{self.id}.m3u8"

    @property
    def saved_name(self):
        return f"{self.id}.{self.extension}"

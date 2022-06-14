import os
import time
from datetime import datetime
from queue import Queue
from threading import Thread

from services.base_service import BaseService


class StorageManager(BaseService):
    def __init__(self):
        super(StorageManager, self).__init__()
        self.buffer = Queue()
        self.worker = Thread(target=self.del_data_job, daemon=True)
        self.worker.start()

    @staticmethod
    def write_to_file(video, file_path):
        with open(file_path, "wb") as file_object:
            file_object.write(video.file.read())

    def remove_later(self, file_path, resampled_video_path, n=0):
        self.buffer.put((file_path, resampled_video_path, datetime.now().timestamp(), n))
        self.logger.info(f"ADD TO REMOVE BUFFER {file_path} + {resampled_video_path}")

    def remove_incremental(self, resampled_video_path):
        if os.path.exists(resampled_video_path):
            os.remove(resampled_video_path)
            i = 0
            while os.path.exists(resampled_video_path.replace(".m3u8", f"{i}.ts")):
                segment_path = resampled_video_path.replace(".m3u8", f"{i}.ts")
                os.remove(segment_path)
                self.logger.info(f"REMOVE SEGMENT {segment_path}")
                i += 1
            return True
        return False

    def del_data_job(self):
        self.logger.info("Start data cleaning job")
        video_path, resampled_video_path, write_time, n = None, None, None, None
        while True:
            try:
                if video_path is None:
                    video_path, resampled_video_path, write_time, n = self.buffer.get()
                if n > 5:
                    self.logger.info(f"NO FILE TO DELETE {video_path} + {resampled_video_path}")
                    continue
                self.logger.info(f"WAIT TO DELETE {video_path} - "
                                 f"{self.config.original_video_time_alive - (datetime.now().timestamp() - write_time)}s"
                                 f" remains")
                duration = datetime.now().timestamp() - write_time
                if duration >= self.config.original_video_time_alive:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    done_remove = self.remove_incremental(resampled_video_path)
                    if done_remove:
                        self.logger.info(f"REMOVE {video_path}")
                    else:
                        self.remove_later(video_path, resampled_video_path, n + 1)
                    video_path = resampled_video_path = write_time = n = None
                else:
                    sleep_time = self.config.original_video_time_alive - (datetime.now().timestamp() - write_time)
                    if sleep_time > 0:
                        self.logger.info(f"Sleep in {sleep_time}s")
                        time.sleep(sleep_time)
            except Exception as e:
                self.logger.error(f"FAILED DELETE VIDEO. Error: {e}")

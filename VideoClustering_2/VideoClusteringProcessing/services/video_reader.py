import time
import traceback
import logging

import imutils
from imutils.video import FileVideoStream
import threading

from objects.video import Video
from config.config import Config
from objects.singleton import Singleton
from services.buffer_manager import BufferManager
from common.queue_name import QName


class VideoReader(metaclass=Singleton):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()
        self.buffer_manager = BufferManager()
        self.queue_name = QName.MESSAGE_VIDEO_Q
        self.target_queue_name = QName.PROCESS_VIDEO_Q
        self.workers = [threading.Thread(target=self.job, daemon=True, args=(thread_id,)) for thread_id in
                        range(self.config.number_video_readers)]
        self.logger.info(f"Create {len(self.workers)} workers")

    def start(self):
        [worker.start() for worker in self.workers]

    def join(self):
        [self.buffer_manager.qs[queue].join() for queue in self.buffer_manager.qs]

    def read_video(self, video: Video) -> Video:
        cap = FileVideoStream(video.url).start()
        count2 = 0
        start_time = time.time()
        while cap.running():
            frame = cap.read()
            try:
                frame.any
            except Exception as e:
                self.logger.error(f"Cannot read frame or EOF. Error: {e}")
                break
            if count2 % self.config.skip_frame == 0:
                h, w, _ = frame.shape
                w = int(w*self.config.image_height/h)
                frame = imutils.resize(frame, w, h)
                h, w, _ = frame.shape
                video.add_frame(org_img=frame)
                if cap.Q.qsize() < 2:  # If we are low on frames, give time to producer
                    time.sleep(0.001)  # Ensures producer runs now, so 2 is sufficient
            count2 += 1
        cap.stop()
        if count2 != 0:
            self.logger.info("---------Load Video Done: {:.2f} s".format(time.time() - start_time))
        return video

    def job(self, thread_id):
        self.logger.info(f"Start thread {thread_id}")
        while True:
            video: Video = self.buffer_manager.get_data(queue_name=self.queue_name)
            try:
                video = self.read_video(video)
                if len(video.frames) == 0:
                    video.status = "DONE"
                    self.buffer_manager.put_data(queue_name=QName.ADD_DB_Q, data=video.info)
                    self.logger.info(f"Video {video.url} has length = 0")
                else:
                    video.status = 'PROCESSING'
                    self.buffer_manager.put_data(queue_name=QName.ADD_DB_Q, data=video.info)
                    self.buffer_manager.put_data(queue_name=self.target_queue_name, data=video)
            except Exception as e:
                video.status = 'FAILED'
                self.buffer_manager.put_data(queue_name=QName.ADD_DB_Q, data=video.info)
                error_msg = str(traceback.format_exc())
                self.logger.error(f'Video Reader Error | Url: {video.url}\n{error_msg} {e}')
                video.delete()


# if __name__ == "__main__":
#     video_reader = VideoReader()
#     video_reader.start()
#     for i in range(10):
#         video_reader.buffer_manager.put_data(queue_name=QName.MESSAGE_VIDEO_Q, data=Video(
#             url='/AIHCM/ComputerVision/tienhn/Video_representation/video/shortvideo1.mp4'))
#     video_reader.join()
#     print(video_reader.buffer_manager.qs['MESSAGE_VIDEO_Q'].qsize(),
#           video_reader.buffer_manager.qs['PROCESS_VIDEO_Q'].qsize())

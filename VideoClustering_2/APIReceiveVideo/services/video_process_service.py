import os
import subprocess
import time

from common.video_status import VideoStatus
from objects.video import Video
from services.base_service import BaseService
from services.storage_manager import StorageManager


class VideoProcessor(BaseService):
    def __init__(self):
        super(VideoProcessor, self).__init__()
        self.storage_manager = StorageManager()

    def __call__(self, video: Video):
        self.logger.info(f"START processing video {video.name}")
        start = time.time()

        resampled_video_path = os.path.join(self.config.resampled_video_folder, video.m3u8)
        original_file_path = os.path.join(self.config.original_video_folder, video.saved_name)

        self.storage_manager.write_to_file(video, original_file_path)

        try:
            result: subprocess.CompletedProcess = subprocess.run(
                ['ffmpeg', '-i', original_file_path, '-profile:v', 'baseline', '-start_number', '0', '-hls_time', '1',
                 '-hls_list_size', '0', '-f', 'hls', '-filter:v', 'fps=60, setpts=PTS/8', '-tune',
                 'fastdecode', '-an', '-preset', 'ultrafast', resampled_video_path], stdout=subprocess.PIPE)

            success = result.returncode == 0
            error = None
            if success:
                video.status = VideoStatus.SUCCESS
            else:
                video.status = VideoStatus.FAILED
        except Exception as e:
            success = False
            error = e
            video.status = VideoStatus.FAILED
        video.sampling_time = int((time.time() - start) * 1000)
        self.logger.info(f"{'DONE' if success else 'FAILED'} processing video {video.name} "
                         f"in {time.time() - start} (s)." + (f"Error {error}" if not success else ""))
        self.storage_manager.remove_later(original_file_path, resampled_video_path)
        return video

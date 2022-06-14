
from objects.video import Video
from services.base_service import BaseService
from services.manager_service import ManagerService
from services.video_process_service import VideoProcessor


class VideoProcessPipeline(BaseService):
    def __init__(self):
        super(VideoProcessPipeline, self).__init__()
        self.manager_service = ManagerService()
        self.video_processor = VideoProcessor()

    def __call__(self, video: Video) -> Video:
        self.logger.info(f"START processing video {video.name}")
        video = self.video_processor(video)
        if video.is_success:
            self.manager_service.update_video(video)
        return video

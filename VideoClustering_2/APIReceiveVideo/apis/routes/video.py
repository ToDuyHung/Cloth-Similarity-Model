from fastapi import UploadFile, File

from apis.routes.base_route import BaseRoute
from pipeline.video_processor import VideoProcessPipeline
from objects.video import Video


class VideoRoute(BaseRoute):
    def __init__(self):
        super(VideoRoute, self).__init__(prefix="/video")
        self.video_processing_pipeline = VideoProcessPipeline()

    def create_routes(self):
        router = self.router

        @router.post("/upload")
        async def send_video(video: UploadFile = File('file')):
            video = Video(video)
            video = await self.wait(self.video_processing_pipeline, video)
            return video.id

        @router.get("/")
        async def root():
            return {"message": "Hello World"}


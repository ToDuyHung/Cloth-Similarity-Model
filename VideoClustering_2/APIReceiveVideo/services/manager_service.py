from services.base_service import BaseServiceSingleton
from objects.video import Video


class ManagerService(BaseServiceSingleton):
    def __init__(self):
        super(ManagerService, self).__init__()
        self.session = None

    def update_video(self, video: Video):
        def update_request():
            video_url = self.config.media_server_api + video.m3u8
            return self.session.post(self.config.manager_api, json={"id": video.id, "videoUrl": video_url,
                                                                    "samplingTime": video.sampling_time,
                                                                    "createdTime": video.created_time}).json()

        result = self.make_request(update_request)
        if result is None:
            self.logger.error(f"FAILED sending video info to API server")
        return result

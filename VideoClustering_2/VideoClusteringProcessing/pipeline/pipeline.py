from services.video_process import VideoProcess
from services.video_reader import VideoReader
from services.message import Message


class PipeLine:
    
    def __init__(self):
        self.video_process = VideoProcess()
        self.video_reader = VideoReader()
        self.message = Message()
        
    def start(self):
        self.video_process.start()
        self.video_reader.start()
        self.message.start()
        
    def join(self):
        self.video_process.join()
        self.video_reader.join()
        self.message.join()

    
if __name__ == "__main__":
    pipeline = PipeLine()
    pipeline.start()
    pipeline.join()
        
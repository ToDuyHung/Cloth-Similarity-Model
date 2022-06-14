from fastapi import FastAPI

from apis.routes.video import VideoRoute


app = FastAPI()
app.include_router(VideoRoute().router)



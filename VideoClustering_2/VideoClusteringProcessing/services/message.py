import json
import logging
import traceback
import requests
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import cv2
from kafka import KafkaConsumer, KafkaProducer

from objects.singleton import Singleton
from services.buffer_manager import BufferManager
from config.config import Config
from common.queue_name import QName
from objects.video import Video


class Message(metaclass=Singleton):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()
        self.buffer_manager = BufferManager()
        self.consumer = KafkaConsumer(self.config.kafka_topic, bootstrap_servers=self.config.bootstrap_servers,
                                      auto_offset_reset=self.config.auto_offset_reset, group_id=self.config.group_id)
        self.producer = KafkaProducer(bootstrap_servers=self.config.bootstrap_servers)
        self.worker = Thread(target=self.put_result, daemon=True)
        self.save_image_worker = Thread(target=self.save_image_job, daemon=True)
        self.consume_video_link_worker = Thread(target=self.consume_video_link_job, daemon=True)

    def start(self):
        self.worker.start()
        self.save_image_worker.start()
        self.consume_video_link_worker.start()

    def consume_video_link_job(self):
        write_log = True
        while True:
            try:
                for message in self.consumer:
                    message = message.value.decode('ascii')
                    message = message.replace("'", '"')
                    message = json.loads(message)
                    self.logger.info(f'INPUT MESSAGE: {message["id"]}')
                    self.buffer_manager.put_data(queue_name=QName.MESSAGE_VIDEO_Q,
                                                 data=Video(url=message['videoUrl'], video_id=message['id']))
                    write_log = True
            except Exception as e:
                if write_log:
                    error_msg = str(traceback.format_exc())
                    self.logger.error(f'Input Message Error: {error_msg}. {e}')
                    self.logger.info('Restarting Consumer...')
                    write_log = False

    def put_result(self):
        while True:
            try:
                video_info = self.buffer_manager.get_data(queue_name=QName.ADD_DB_Q)
                self.producer.send(self.config.kafka_db_topic, value=json.dumps(video_info).encode('utf-8'))
                video_url = video_info["videoUrl"]
                self.logger.info(f'SUCCESS PUT RESULT: {video_url}')
            except Exception as e:
                error_msg = str(traceback.format_exc())
                self.logger.error(f'ERROR WHEN PUT RESULT: {error_msg} {e}')

    def save_image_job(self):
        def job(image, save_image_url):
            try:
                _, im_buf_arr = cv2.imencode(".jpg", image)
                byte_im = im_buf_arr.tobytes()
                url = self.config.file_server_url
                payload = {'IsUseDefaultName': 'true'}
                files = [('FileContent', (save_image_url, byte_im, 'image/jpeg'))]
                headers = {}
                response = None
                while response is None:
                    try:
                        response = requests.request("POST", url, headers=headers, data=payload, files=files)
                        time.sleep(1)
                    except Exception as e:
                        self.logger.error(f"Failed sending Image {save_image_url} to file server. Error: {e}")
                self.logger.info(f'SUCCESS SAVE IMAGE : {save_image_url}')
            except Exception as e:
                error_msg = str(traceback.format_exc())
                self.logger.error(f'ERROR SAVE IMAGE: {error_msg} {e}')

        executor = ThreadPoolExecutor(max_workers=10)
        self.logger.info("START SAVE IMAGE JOB")
        while True:
            img, image_url = self.buffer_manager.get_data(queue_name=QName.OUT_IMAGE_Q)
            executor.submit(job, img, image_url)

    def join(self):
        self.worker.join()
        self.save_image_worker.join()
        self.consume_video_link_worker.join()

import os
import json

from common.common_keys import *
from objects.singleton import Singleton


class Config(metaclass=Singleton):
    thresh_cluster_l1 = float(os.getenv(THRESH_CLUSTER_L1, 0.978))
    crop_ratio = float(os.getenv(CROP_RATIO, -0.08))
    thresh_triplet_model_l1 = float(os.getenv(THRESH_TRIPLET_MODEL_L1, 0.82))
    thresh_triplet_model_l2 = float(os.getenv(THRESH_TRIPLET_MODEL_L2, 0.89))
    skip_frame = int(os.getenv(SKIP_FRAME, 4))
    number_of_workers = int(os.getenv(NUMBER_OF_WORKER, 1))
    number_video_readers = int(os.getenv(NUMBER_VIDEO_READERS, 2))
    max_svd_dimension = int(os.getenv(MAX_SVD_DIMENSION, 63))
    output_fps = int(os.getenv(OUTPUT_FPS, 10))
    output_video = os.getenv(OUTPUT_VIDEO, 'output_video/Output.avi')
    kafka_topic = os.getenv(KAFKA_TOPIC, 'test_model_tiens')
    kafka_db_topic = os.getenv(KAFKA_DB_TOPIC, 'clusterResult')
    bootstrap_servers = os.getenv(BOOTSTRAP_SERVERS, '172.29.13.24:35000')
    auto_offset_reset = os.getenv(OFFSET_RESET, 'earliest')
    group_id = os.getenv(GROUP_ID, 'video_representations')
    log_file = os.getenv(LOG_FILE, 'logs/app.log')
    main_logger_name = os.getenv(MAIN_LOGGER_NAME, 'ChatbotVideo')
    perf_log_file = os.getenv(PERF_LOG_FILE, 'logs/performance.log')
    perf_logger_name = os.getenv(PERFORMANCE_LOGGER_NAME, 'Performance_logging')
    file_server_url = os.getenv(FILE_SERVER_URL, 'http://172.28.0.23:35432/api/file/upload-file-local')
    write_output = bool(int(os.getenv(WRITE_OUTPUT, True)))
    file_server_get_url = os.getenv(GET_URL, 'http://172.28.0.23:35432/api/file/Get-File-Local?guid=')
    image_height = int(os.getenv(IMAGE_HEIGHT, 320))
    cloth_model_path = os.getenv(CLOTH_MODEL, 'model/cloth_model')
    use_gpu = bool(int(os.getenv(USE_GPU, False)))
    crop_model_path = os.getenv(CROP_MODE, 'model/yolov5/yolov5s.pt')
    crop_conf = os.getenv(CROP_CONF, 0.92)
    

    def __repr__(self):
        return json.dumps({key: getattr(self, key)
                           for key in self.__dir__() if "__" != key[:2] and "__" != key[-2:] and key != "dict"}
                          , indent=4)

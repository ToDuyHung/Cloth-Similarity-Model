import json
import logging
import time
from typing import List
import sys

import requests
import numpy as np
import cv2

from common.common_keys import *
from objects.frame import Frame
from objects.cluster import Cluster
from config.config import Config


class Video:
    
    def __init__(self, video_id: str = 'test', url: str = None, frames: List[Frame] = None,
                 clusters: List[Cluster] = None, number_frame_l1_cluster: int = None, skip_frame = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.url = url
        self.video_id = video_id
        self.frames = frames if frames is not None else []
        self.clusters = clusters if clusters is not None else []
        self.number_frame_l1_cluster = number_frame_l1_cluster
        self.config = Config()
        self.status = 'CREATED'
        self.skip_frame = None if skip_frame is None else self.config.skip_frame

    def add_frame(self, org_img: np.array):
        frame = Frame(index=len(self), org_img=org_img)
        self.frames.append(frame)
        return frame

    def add_cluster(self, cluster: dict):
        self.clusters = []
        for key, value in cluster.items():
            self.clusters.append(Cluster(video_id=self.video_id,
                                         list_frame=list(range(value[1][0], value[1][1] + 1)),
                                         key_frame=key,
                                         label=value[0]))
    
    def del_frame_not_key(self, cluster: dict):
        key_frames_1 = []
        key_frames_2 = []
        for key in cluster.keys():
            key_frames_1.append(key)
        if len(list(cluster.values())[0]) > 2:
            for key in cluster.values():
                key_frames_2.append(key[2])
        self.frames = [frame for frame in self.frames if frame.id in key_frames_1] + [frame for frame in self.frames if frame.id in key_frames_2]

    @property
    def to_save_images(self):
        output = []
        for index, cluster in enumerate(self.clusters):
            if cluster.label:
                img = self.frames[index].org_img
                image_url = cluster.image_url.split('guid=')[-1]
                output.append((img, image_url))
        return output

    def __len__(self):
        if self.frames is not None:
            return len(self.frames)
        return 0

    def get_items(self):
        dict_img = {}
        for index, cluster in enumerate(self.clusters):
            dict_img[cluster.key_frame] = self.frames[index].image
            dict_img[cluster.key_frame+0.5] = self.frames[index + len(self.clusters)].image
        return dict_img

    def __iter__(self):
        for frame in self.frames:
            yield frame

    @property
    def labels(self):
        labels = []
        count = 1
        for value in self.clusters:
            value = (value.label, [value.list_frame[0], value.list_frame[-1]])
            if value[0] == 0 and (count <= 1 or ((value[1][1] - value[1][0] + 1) <= 2 * self.number_frame_l1_cluster)):
                labels += [''] * (value[1][1] - value[1][0] + 1)
            else:
                queue_label = min(int((value[1][1] - value[1][0] + 1) * 0.08), 10)
                labels += [''] * queue_label + [str(count)] * (value[1][1] - value[1][0] + 1 - queue_label)
                count += 1
        return labels

    @property
    def images(self):
        return [frame.image for frame in self.frames]

    @property
    def info(self):
        return {
            ID: self.video_id,
            VIDEO_URL: self.url,
            STATUS: self.status,
            CLUSTER_INFOS: [cluster.info for cluster in self.clusters if cluster.image_url is not None]
        }

    @property
    def profile(self):
        output = {key: sys.getsizeof(getattr(self, key)) for key in self.__dir__()
                  if "__" not in key and key != "profile"}
        output["frames"] = sum(item.size * item.itemsize for item in self.images)
        output = f"PROFILE {self.video_id}: " + json.dumps(output, indent=4)
        return output

    def delete(self):
        for frame in self.frames:
            frame.delete()

        for cluster in self.clusters:
            cluster.delete()
        self.config = None
        self.frames = None
        self.clusters = None
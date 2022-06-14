import os
from collections import Counter
import traceback
import logging
from typing import Dict, Tuple, List
import time
import json
from memory_profiler import profile

import numpy as np
import cv2
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import webcolors
from matplotlib import pyplot as plt
import threading

from config.config import Config
from utils.model_utils import ClothModel, CropModel
from services.buffer_manager import BufferManager
from common.queue_name import QName
from objects.video import Video
from objects.singleton import Singleton


class VideoProcess(metaclass=Singleton):

    def __init__(self) -> None:
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger_perf = logging.getLogger(self.config.perf_logger_name)
        self.buffer_manager = BufferManager()
        self.workers = [threading.Thread(target=self.job, daemon=True, args=(thread_id,)) for thread_id in
                        range(self.config.number_of_workers)]
        self.logger.info(f"Create {len(self.workers)} workers")
        self.cloth_model = ClothModel()
        self.crop_model = CropModel()

    def start(self):
        [worker.start() for worker in self.workers]

    def join(self):
        [worker.join() for worker in self.workers]

    @staticmethod
    def svd(feature_l1, svd_dimension=63):
        a = csc_matrix(feature_l1, dtype=float)
        u, s, vt = svds(a, k=svd_dimension)
        v1_t = vt.transpose()
        projections = v1_t @ np.diag(s)
        return projections

    def svd_similarity(self, projections, svd_dimension=63):
        f = projections
        c = dict()
        for i in range(f.shape[0]):
            c[i] = np.empty((0, svd_dimension), int)
        c[0] = np.vstack((c[0], f[0]))
        c[0] = np.vstack((c[0], f[1]))
        e = dict()
        for i in range(projections.shape[0]):
            e[i] = np.empty((0, svd_dimension), int)
        e[0] = np.mean(c[0], axis=0)
        count = 0
        for i in range(2, f.shape[0]):
            similarity = np.dot(f[i], e[count]) / ((np.dot(f[i], f[i]) ** .5) * (np.dot(e[count], e[count]) ** .5))
            if similarity < self.config.thresh_cluster_l1:
                count += 1
                c[count] = np.vstack((c[count], f[i]))
                e[count] = np.mean(c[count], axis=0)
            else:
                c[count] = np.vstack((c[count], f[i]))
                e[count] = np.mean(c[count], axis=0)
        return f, c

    @staticmethod
    def reduce_first_cluster(projections, c):
        b = []
        for i in range(projections.shape[0]):
            b.append(c[i].shape[0])
        last = b.index(0)
        b1 = b[:last]
        return b1

    @staticmethod
    def svd_extract_feature(frame_rgb):
        height, width, channels = frame_rgb.shape
        if height % 3 == 0:
            h_chunk = int(height / 3)
        else:
            h_chunk = int(height / 3) + 1

        if width % 3 == 0:
            w_chunk = int(width / 3)
        else:
            w_chunk = int(width / 3) + 1
        h, w = 0, 0
        feature_vector = []
        for a in range(1, 4):
            h_window = h_chunk * a
            for b in range(1, 4):
                frame = frame_rgb[h: h_window, w: w_chunk * b, :]
                hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256])
                hist1 = hist.flatten()
                feature_vector += list(hist1)
                w = w_chunk * b
            h = h_chunk * a
            w = 0
        return feature_vector

    def post_process_first_cluster(self, distribute_label_l1):
        b1_new = []
        tmp = 0
        for i in distribute_label_l1:
            if i == 1:
                tmp += 1
            else:
                if tmp != 0:
                    b1_new.append(tmp)
                    tmp = 0
                b1_new.append(i)
        b1_copy = list(set(b1_new.copy()))
        b1_copy.sort()
        x = np.median(b1_new)
        self.logger.info(str((x, b1_copy[-2])))
        frame_each_item = min(b1_copy[-3], x + 1 / np.log10(x))  # min(len(D)/15, 10)
        self.logger.info(str(frame_each_item))
        b2 = np.zeros(len(distribute_label_l1))
        count = 1
        for k, v in enumerate(distribute_label_l1):
            if v >= frame_each_item:
                b2[k] = count
                count += 1
        labels = [0] * distribute_label_l1[0]
        id_label = 0
        for i in range(1, len(b2)):
            if b2[i] == 0:
                labels = labels + [0] * distribute_label_l1[i]
            elif b2[i] != b2[i - 1]:
                id_label += 1
                labels = labels + [id_label] * distribute_label_l1[i]
            else:
                labels = labels + [id_label] * distribute_label_l1[i]
        self.logger.info("LABELS" + str(labels))
        return labels, frame_each_item

    @staticmethod
    def find_range_value(labels):
        """
        Input: a list
        Output: a dictionary contain the range of each element
        """
        list_point = {}
        for value in list(set(labels)):
            list_begin_end = [-100, -100]
            for idx in range(len(labels)):
                if (labels[idx] == value and labels[idx - 1] != value) or idx == 0:
                    list_begin_end[0] = idx
                elif (labels[idx] != value and labels[idx - 1] == value) or idx == len(labels) - 1:
                    if idx == len(labels) - 1:
                        list_begin_end[1] = idx
                    else:
                        list_begin_end[1] = idx - 1
                if -100 not in list_begin_end:
                    key_2 = int((list_begin_end[1] - list_begin_end[0]) * 3/4 + list_begin_end[0])
                    list_point[int(sum(list_begin_end) / len(list_begin_end))] = (value, list_begin_end, key_2)
                    list_begin_end = [-100, -100]
        list_point = dict(sorted(list_point.items()))
        return list_point

    @staticmethod
    def show_cluster(whole_frame, list_point):
        columns = 5
        rows = len(list_point) // columns + 1
        fig = plt.figure(figsize=(12, 12))
        for k, v in enumerate(list_point):
            img = whole_frame[v].copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.putText(img, f"{list_point[v][0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            fig.add_subplot(rows, columns, k + 1)
            plt.imshow(img)

    @staticmethod
    def display_two_images(img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(4, 4))
        columns = 2
        rows = 1
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img1)
        fig.add_subplot(rows, columns, 2)
        plt.imshow(img2)
        plt.show()

    def second_cluster_batch(self, video: Video) -> Tuple[Video, List[float]]:
        i, image_id = 0, 0
        list_point = {}
        dict_img = video.get_items()
        s = time.time()
        dict_img_feature = self.cloth_model.extract_image_batch(dict_img=dict_img.copy())
        s1 = time.time()
        for index, cluster in enumerate(video.clusters):
            list_point[cluster.key_frame] = (cluster.label, [cluster.list_frame[0], cluster.list_frame[-1]])
        while i < 1:
            self.logger.info(f"----------- Begin Loop ----------- {i}")
            self.logger.info("POINT LIST: " + str(list_point))
            res, res_key = [], []
            key_0 = list(list_point.keys())[0]
            res.append(list_point[key_0])
            res_key.append(key_0)
            flag = False
            for idx in range(1, len(list_point)):
                self.logger.info('++++++++++++ Begin a couple sample ++++++++++++')
                key = list(list_point.keys())[idx]
                cluster, point_range = list_point[key]
                pre_cluster, pre_point_range = res[-1]
                pre_key = res_key[-1]
                self.logger.info(f"Go to compare: {pre_cluster}, {cluster}")
                diff_1 = (cosine_similarity(dict_img_feature[key].reshape(1,-1), dict_img_feature[pre_key].reshape(1,-1)))[0][0]
                diff_2 = (cosine_similarity(dict_img_feature[key+0.5].reshape(1,-1), dict_img_feature[pre_key+0.5].reshape(1,-1)))[0][0]                
                self.logger.info(f'\t\t Model check cluster {pre_cluster} and {cluster}:  {diff_1} {diff_2}')
                cv2.imwrite(f'output_video/image/{i}_{key}.jpg', dict_img[key])
                cv2.imwrite(f'output_video/image/{i}_{key+0.5}.jpg', dict_img[key+0.5])
                cv2.imwrite(f'output_video/image/{i}_{pre_key}.jpg', dict_img[pre_key])
                cv2.imwrite(f'output_video/image/{i}_{pre_key+0.5}.jpg', dict_img[pre_key+0.5])
                self.logger.info(f'\t\t diff_1: {i}_{key}.jpg and {i}_{pre_key}.jpg')
                self.logger.info(f'\t\t diff_2: {i}_{key+0.5}.jpg and {i}_{pre_key+0.5}.jpg')
                if diff_1 >= self.config.thresh_triplet_model_l2 or diff_2 >= self.config.thresh_triplet_model_l2 or (diff_1 >= self.config.thresh_triplet_model_l1 and diff_2 >= self.config.thresh_triplet_model_l1):
                    flag = True
                    if pre_cluster != 0:
                        new_cluster = pre_cluster
                        new_key = pre_key
                    else:
                        new_cluster = cluster
                        new_key = key
                    new_point_range = [pre_point_range[0], point_range[1]]
                    res.pop()
                    res.append((new_cluster, new_point_range))
                    res_key.pop()
                    res_key.append(new_key)
                    self.logger.info(f'\t\t Combine cluster {pre_cluster} and {cluster}')
                    continue
                res.append(list_point[key])
                res_key.append(key)
            res_dict = {}
            for index, value in enumerate(res):
                res_dict[res_key[index]] = value
            res_dict = dict(sorted(res_dict.items()))
            if not flag:
                self.logger.info('Break')
                break
            else:
                list_point = res_dict
            i += 1
        count = 1
        for key, value in list_point.items():
            if value[0] == 0 and ((value[1][1] - value[1][0] + 1) <= 2 * video.number_frame_l1_cluster):
                item_id = 0
            else:
                item_id = count
                count += 1
            list_point[key] = (item_id, [value[1][0], value[1][1]])
        video.add_cluster(list_point)
        video.del_frame_not_key(list_point)
        self.logger.info(str(list_point))
        list_time = [s1 - s]
        return video, list_time

    def write_output(self, video: Video, fps=10, output_path='output_video/Output.avi'):
        if not os.path.exists('output_video'):
            os.mkdir('output_video')
        count_frame = 0
        labels = video.labels
        dict_label = {}
        for i in range(len(labels)):
            if i > 0:
                dict_label[i * self.config.skip_frame] = labels[count_frame]
                count_frame += 1
        with open(os.path.join('./output_video', video.url.split('/')[-1][:-4] + '_predict.json'), 'w') as f:
            json.dump(dict_label, f)

    def extract_feature_l1(self, video: Video = None):
        arr = np.empty((0, 1944), int)
        for idx in range(len(video.frames)):
            frame = video.frames[idx].org_img
            feature_vector = self.svd_extract_feature(frame)
            video.frames[idx].feature = feature_vector
            arr = np.vstack((arr, feature_vector))
        feature_l1 = arr.transpose()
        return feature_l1, video

    @profile
    def first_cluster(self, video: Video):
        self.logger.info(f"VIDEO LENGTH {id(video)}, {len(video.frames)}, {video.profile}")
        feature_l1, video = self.extract_feature_l1(video=video)
        svd_dimension = min(self.config.max_svd_dimension, feature_l1.shape[1] - 1)
        projections = self.svd(feature_l1, svd_dimension=svd_dimension)
        projections, c = self.svd_similarity(projections=projections, svd_dimension=svd_dimension)
        distribute_label_l1 = self.reduce_first_cluster(projections=projections, c=c)
        labels, frame_l1_cluster = self.post_process_first_cluster(distribute_label_l1=distribute_label_l1)
        video.number_frame_l1_cluster = frame_l1_cluster
        list_point = self.find_range_value(labels)
        video.add_cluster(list_point)
        video.del_frame_not_key(list_point)
        return video

    def job(self, thread_id):
        self.logger.info(f'Working on {thread_id}')
        while True:
            video: Video = self.buffer_manager.get_data(queue_name=QName.PROCESS_VIDEO_Q)
            try:
                s = time.time()
                video = self.first_cluster(video=video)
                s1 = time.time()
                video, list_time = self.second_cluster_batch(video=video)
                s2 = time.time()
                output_video = self.config.output_video
                count = 0
                if self.config.write_output:
                    while os.path.exists(output_video):
                        output_video = self.config.output_video.split('.')[0] + str(count) + '.' + \
                                       self.config.output_video.split('.')[1]
                        count += 1
                    self.write_output(video=video, fps=self.config.output_fps, output_path=output_video)
                video.status = 'DONE'
                self.buffer_manager.put_data(queue_name=QName.ADD_DB_Q, data=video.info)
                [self.buffer_manager.put_data(queue_name=QName.OUT_IMAGE_Q, data=item) for item in video.to_save_images]
                self.logger_perf.info(f'\n\n\nUrl: {video.url}')
                time.sleep(0.02)
                self.logger_perf.info('%15s%15s%15s%15s%15s' % ('Total frame', 'Total time', 'First cls',
                                                                        'Second cls', 'Model predict'))
                self.logger_perf.info('%15s%15s%15s%15s%15s' % (f'{len(video)}', f'{round(time.time() - s, 2)}',
                                                                        f'{round(s1 - s, 2)}', f'{round(s2 - s1, 2)}',
                                                                        f'{round(list_time[0], 2)}'))
            except Exception as e:
                video.status = 'FAILED'
                self.buffer_manager.put_data(queue_name=QName.ADD_DB_Q, data=video.info)
                error_msg = str(traceback.format_exc())
                self.logger.error(f'Video Process Error | Url: {video.url}\n{error_msg} {e}')
            video.delete()

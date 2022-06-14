import numpy as np


class Frame:
    def __init__(self, index, org_img: np.array = None, feature: np.array = None):
        self.id = index
        # self.image = image
        self.feature = feature  # level 1
        self.org_img = org_img

    @property
    def image(self):
        h, w, c = self.org_img.shape
        frame = self.org_img[int(h * 0.1):int(h * 0.9), int(w * 0.15):int(w * 0.85)]
        return frame

    def delete(self):
        self.org_img = None
        self.feature = None

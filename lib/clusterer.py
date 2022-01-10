import os
from lib.image_matrix import VGGMatrix
import cv2

class Clusterer:
    def __init__(self, image_dir_path, threshold = 10):
        '''
        If the image distance is gretter than the
        threshold than its properly a different
        cluster.

        '''
        self._load_images(image_dir_path = image_dir_path)
        self.image_matrix = VGGMatrix()
        self.threshold = threshold
        self.cluster_data = []


    def _load_images(self, image_dir_path):
        filenames = os.listdir(image_dir_path)
        self.images = []

        for filename in filenames:
            path = os.path.join(image_dir_path, filename)
            image = cv2.imread(path)
            self.images.append(image)


    def run(self):
        pass




import os
from lib.image_matrix import VGGMatrix
from lib.utils import euclidean_dist
import cv2

class Clusterer:
    def __init__(self, image_dir_path, threshold = 10, max_image_per_dir = 10):
        '''
        threshold : If the image distance is gretter than the
        threshold than its properly a different
        cluster.

        max_image_per_dir : number of image we will take
        from each directory to compare

        '''
        self._load_images(image_dir_path = image_dir_path)
        self.image_matrix = VGGMatrix()
        self.threshold = threshold
        self.cluster_data = []
        self.max_image_per_dir = max_image_per_dir


    def _load_images(self, image_dir_path):
        filenames = os.listdir(image_dir_path)
        self.images = []

        for filename in filenames:
            path = os.path.join(image_dir_path, filename)
            image = cv2.imread(path)
            self.images.append(image)


    def run(self):
        for image in self.images:
            self._add(image = image)


    def _add(self, image):
        closest_neighbore_index = self._get_closest_neighbore(image = image)


        if closest_neighbore_index == -1:
            self.cluster_data.append([image])
            return

        self.cluster_data[closest_neighbore_index].append(image)

    def _get_closest_neighbore(self, image):

        if len(self.cluster_data) == 0:
            return -1

        current_matrix = self.image_matrix.get_matrix(image = image)
        index = self._get_nearest_index(matrix = current_matrix)

        return index

    def _get_nearest_index(self, matrix):
        nearest_distance = 100000
        nearest_index = -1

        for current_index, image_list in enumerate(self.cluster_data):
            for image in image_list[:self.max_image_per_dir]:
                current_matrix = self.image_matrix.get_matrix(image = image)
                current_distance = euclidean_dist(x = matrix,y =  current_matrix)

                if current_distance < nearest_distance:
                    nearest_distance = current_distance
                    nearest_index = current_index


        if nearest_distance > self.threshold:
            nearest_index = -1

        return nearest_index

    def save(self, save_dir = None):

        if save_dir is None:
            save_dir = f"threshold_{self.threshold}"

        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)


        for index, image_list in enumerate(self.cluster_data):
            current_cluster_path = os.path.join(save_dir, f"{index}")

            if os.path.isdir(current_cluster_path) == False:
                os.mkdir(current_cluster_path)

            for cluster_image in  image_list:
                index = len(os.listdir(current_cluster_path))
                save_image_name = f"{index}.jpg"
                save_cluster_image_path = os.path.join(current_cluster_path, save_image_name)
                cv2.imwrite(save_cluster_image_path, cluster_image)















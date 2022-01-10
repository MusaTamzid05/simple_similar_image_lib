from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
import cv2
import numpy as np



class VGGMatrix:
    def __init__(self, size = 224):
        self.size = size
        self._load_model()


    def _load_model(self):
        self.model = VGG16(
                weights = "imagenet",
                include_top = False,
                input_tensor = Input(shape = (self.size, self.size, 3))
                )


    def get_matrix(self, image):
        if type(image) == str:
            image_path = image
            image = cv2.imread(image)

            if image is None:
                raise RuntimeError(f"{image_path} could not be loaded.")

        resized_image = cv2.resize(image, (self.size, self.size))
        query_image = np.expand_dims(resized_image, 0)
        result = self.model.predict(query_image)

        output_shape = result.shape

        result = result.reshape((1, output_shape[1] * output_shape[2] * output_shape[3]))
        result /= 255.0

        return result








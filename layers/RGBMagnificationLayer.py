import numpy as np
import cv2
import matplotlib.pyplot as plt

from Layer import PreprocessLayer

class RGBMagnificationLayer(PreprocessLayer):
    def __init__ (self, in_dim, out_dim, channel):
        super().__init__(size=in_dim, name="rgbMagnification")
        self.in_dim = in_dim
        self.out_dim = out_dim
        if not (channel == 'R' or channel == 'G' or channel == 'B'):
            raise Exception(f"Channel must be either: 'R', 'G', or 'B' not '{channel}'")
        else:
            self.offset = 0 if channel == 'R' else 1 if channel == 'G' else 2

    def process(self, image):
        original_frame = image
        inverted_image = cv2.bitwise_not(original_frame)

        original_array = np.array(original_frame, dtype=np.int64)
        inv_original_array = np.array(inverted_image, dtype=np.int64)

        channel = original_array[:, :, (2 + self.offset)%3]
        result_image = original_array.copy()

        CHANGE_CHANNEL = (0 + self.offset)%3

        # MIN MAX NORM
        result_image[:, :, CHANGE_CHANNEL] = (
            (result_image[:, :, CHANGE_CHANNEL] - np.min(result_image[:, :, CHANGE_CHANNEL]))
            / (np.max(result_image[:, :, CHANGE_CHANNEL]) - np.min(result_image[:, :, CHANGE_CHANNEL]))
            * 255
        )

        channel = result_image[:, :, (2 + self.offset)%3]

        result_image[:, :, CHANGE_CHANNEL] = channel * (
            channel + inv_original_array[:, :, CHANGE_CHANNEL]
        )
        result_image[:, :, (1 + self.offset)%3] = 0
        result_image[:, :, (2 + self.offset)%3] = 0

        # MIN MAX NORM
        result_image[:, :, CHANGE_CHANNEL] = (
            (result_image[:, :, CHANGE_CHANNEL] - np.min(result_image[:, :, CHANGE_CHANNEL]))
            / (np.max(result_image[:, :, CHANGE_CHANNEL]) - np.min(result_image[:, :, CHANGE_CHANNEL]))
            * 255
        )

        result_image = np.clip(result_image, 0, 255)
        result_image = result_image.astype(np.uint8)
        return (result_image, None)
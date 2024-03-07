import cv2
import numpy as np

def preprocess_frame(self, frame):
    """
    Extracts the red channel from the frame and magnifies its values.
    """
    original_frame = frame
    inverted_image = cv2.bitwise_not(original_frame)

    original_array = np.array(original_frame, dtype=np.int64)
    inv_original_array = np.array(inverted_image, dtype=np.int64)

    red_channel = original_array[:, :, 2]
    result_image = original_array.copy()

    # MIN MAX NORM
    result_image[:, :, 0] = (
        (result_image[:, :, 0] - np.min(result_image[:, :, 0]))
        / (np.max(result_image[:, :, 0]) - np.min(result_image[:, :, 0]))
        * 255
    )

    red_channel = result_image[:, :, 2]

    result_image[:, :, 0] = red_channel * (
        red_channel + inv_original_array[:, :, 0]
    )
    result_image[:, :, 1] = 0
    result_image[:, :, 2] = 0

    # MIN MAX NORM
    result_image[:, :, 0] = (
        (result_image[:, :, 0] - np.min(result_image[:, :, 0]))
        / (np.max(result_image[:, :, 0]) - np.min(result_image[:, :, 0]))
        * 255
    )

    result_image = np.clip(result_image, 0, 255)
    result_image = result_image.astype(np.uint8)

    self.image_pub_pre.publish(np.array(result_image))

    return result_image

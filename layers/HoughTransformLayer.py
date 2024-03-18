import cv2
import numpy as np

from Layer import AnalysisLayer

class HoughTransformLayer(AnalysisLayer):

    def __init__(self, size, threshold, pass_post_processing_img = False) -> None:
        """
        Apply Hough Transform for line detection on the given image.

        Parameters:
            threshold (int): Threshold value for line detection. Defaults to 100.

        Returns:
            lines: A list of lines detected in the image, each line represented by rho and theta values.
        """
        super().__init__(size)
        self.threshold = threshold
        self.pass_post_processing_img = pass_post_processing_img

    def process(self, image):

        lines = cv2.HoughLines(image, rho=1, theta=np.pi/180, threshold=self.threshold)
        image_with_lines = image.copy()
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return (image_with_lines if self.pass_post_processing_img else image, lines)
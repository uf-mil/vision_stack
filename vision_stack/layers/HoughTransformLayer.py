import cv2
import numpy as np
import math

from .Layer import AnalysisLayer

try:
    from mil_msgs.msg import Lines, HoughTransform
except Exception as e:
    print(f"Could not import messages because:\n{e}")

class HoughTransformLayer(AnalysisLayer): #TODO: Write a more detailed description for layer

    def __init__(self, threshold, min_line_length, max_line_gap, pass_post_processing_img = False) -> None:
        """
        Apply Hough Transform for line detection on the given image.

        Parameters:
            threshold (int): Threshold value for line detection. Defaults to 100.

        Returns:
            lines: A list of lines detected in the image, each line represented by rho and theta values.
        """
        super().__init__(name="houghTransform")
        self.threshold = threshold
        self.pass_post_processing_img = pass_post_processing_img
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.msg = None

    def process(self, image):

        edges = cv2.Canny(image, 20, 200)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_image = np.zeros_like(edges)

        contour_image_bgr = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(contour_image, contours, -1, (255), 2)

        lines = cv2.HoughLinesP(
            contour_image,
            1,
            np.pi / 180,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        lines_msgs = []

        if lines is not None:
            for i in range(len(lines)):
                line = lines[i]
                x1, y1, x2, y2 = line[0]

                length, center_x, center_y, angle = self.calculate_properties(x1,y1,x2,y2)
                lines_msgs.append(HoughTransform(length = length, center_x=center_x, center_y=center_y, angle=angle))

                cv2.line(
                    contour_image_bgr,
                    (x1, y1),
                    (x2, y2),
                    (255 * (i / len(lines)), 255 - (255 * (i / len(lines))), 0),
                    2,
                )
        
        self.msg = Lines(lines=lines_msgs)

        return (contour_image_bgr if self.pass_post_processing_img else image, [] if lines is None else lines)
    
    def calculate_properties(self, x1, y1, x2, y2):
        # Calculate the length
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Calculate the center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate the angle
        if y2 > y1:  # if y2 is above y1
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        else:  # if y1 is above y2
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        
        # Normalize the angle to the range [0, 180]
        if angle < 0:
            angle += 180
        
        return length, center_x, center_y, angle

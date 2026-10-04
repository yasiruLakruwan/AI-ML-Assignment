import cv2
import numpy as np


class BedRegion:

    def __init__(self, polygon):
        self.polygon = np.array(
            polygon,
            dtype=np.int32
        )

    def contains(self, point):

        x, y = point

        result = cv2.pointPolygonTest(
            self.polygon,
            (float(x), float(y)),
            False
        )

        return result >= 0

    def center(self):

        moments = cv2.moments(self.polygon)

        if moments["m00"] == 0:
            return (0, 0)

        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]

        return (x, y)
    
import math
import numpy as np


class StateClassifier:

    def __init__(self, bed_region):

        self.bed_region = bed_region

        self.previous_center = None
        self.previous_timestamp = None

    def calculate_angle(
        self,
        shoulder,
        hip
    ):

        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]

        angle = math.degrees(
            math.atan2(abs(dx), abs(dy))
        )

        return angle

    def calculate_speed(
        self,
        center,
        timestamp
    ):

        if (
            self.previous_center is None
            or self.previous_timestamp is None
        ):
            self.previous_center = center
            self.previous_timestamp = timestamp

            return 0.0

        distance = math.sqrt(
            (center[0] - self.previous_center[0]) ** 2 +
            (center[1] - self.previous_center[1]) ** 2
        )

        dt = timestamp - self.previous_timestamp

        self.previous_center = center
        self.previous_timestamp = timestamp

        if dt <= 0:
            return 0

        return distance / dt

    def classify(
        self,
        observation,
        timestamp
    ):

        keypoints = np.array(
            observation["keypoints"]
        )

        shoulder = (
            keypoints[5] +
            keypoints[6]
        ) / 2

        hip = (
            keypoints[11] +
            keypoints[12]
        ) / 2

        knee = (
            keypoints[13] +
            keypoints[14]
        ) / 2

        center = observation["center"]

        body_angle = self.calculate_angle(
            shoulder,
            hip
        )

        speed = self.calculate_speed(
            center,
            timestamp
        )

        on_bed = self.bed_region.contains(
            hip
        )

        bbox = observation["bbox"]

        height = bbox[3] - bbox[1]

        if height < 50:
            return {
                "state": "UNKNOWN",
                "confidence": 0.20
            }

        # ----------------------------------
        # LYING
        # ----------------------------------

        if on_bed and body_angle > 55:

            return {
                "state": "LYING_IN_BED",
                "confidence": 0.90
            }

        # ----------------------------------
        # SITTING
        # ----------------------------------

        if 25 <= body_angle <= 55:

            if on_bed:

                return {
                    "state": "SITTING_ON_BED",
                    "confidence": 0.80
                }

            else:

                return {
                    "state": "SITTING_OUTSIDE_BED",
                    "confidence": 0.75
                }

        # ----------------------------------
        # STANDING
        # ----------------------------------

        if body_angle < 25:

            if speed > 120:

                return {
                    "state": "WALKING",
                    "confidence": 0.80
                }

            return {
                "state": "STANDING",
                "confidence": 0.85
            }

        return {
            "state": "UNKNOWN",
            "confidence": 0.30
        }
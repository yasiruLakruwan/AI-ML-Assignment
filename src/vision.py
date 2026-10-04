import math
import numpy as np

from ultralytics import YOLO


class PersonTracker:

    def __init__(self):

        self.model = YOLO("yolo26n-pose.pt")

        self.target_id = None
        self.last_center = None

    def _distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def process(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.25
        )

        if not results:
            return None

        result = results[0]

        if result.boxes is None:
            return None

        if result.keypoints is None:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()

        keypoints = result.keypoints.xy.cpu().numpy()

        # Track IDs may not exist temporarily
        if result.boxes.id is not None:

            ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

        else:

            ids = np.arange(len(boxes))

        if len(boxes) == 0:
            return None

        # ------------------------------------------------
        # Calculate centers
        # ------------------------------------------------

        centers = []

        for box in boxes:

            x1, y1, x2, y2 = box

            center = (
                (x1 + x2) / 2,
                (y1 + y2) / 2
            )

            centers.append(center)

        # ------------------------------------------------
        # Choose target person
        # ------------------------------------------------

        selected_index = None

        # CASE 1:
        # We already know the person's tracking ID.
        if self.target_id is not None:

            matching = np.where(
                ids == self.target_id
            )[0]

            if len(matching) > 0:

                selected_index = int(
                    matching[0]
                )

        # CASE 2:
        # Tracking ID disappeared.
        # Try to find the person closest to
        # their previous position.
        if selected_index is None:

            if self.last_center is not None:

                distances = [
                    self._distance(
                        center,
                        self.last_center
                    )
                    for center in centers
                ]

                selected_index = int(
                    np.argmin(distances)
                )

            else:

                # First detection:
                # choose the largest person.
                areas = []

                for box in boxes:

                    x1, y1, x2, y2 = box

                    area = (
                        (x2 - x1)
                        *
                        (y2 - y1)
                    )

                    areas.append(area)

                selected_index = int(
                    np.argmax(areas)
                )

        # ------------------------------------------------
        # Update target ID
        # ------------------------------------------------

        self.target_id = int(
            ids[selected_index]
        )

        self.last_center = centers[
            selected_index
        ]

        # ------------------------------------------------
        # Return observation
        # ------------------------------------------------

        return {
            "track_id": self.target_id,
            "bbox": boxes[
                selected_index
            ].tolist(),

            "keypoints": keypoints[
                selected_index
            ].tolist(),

            "center": centers[
                selected_index
            ]
        }
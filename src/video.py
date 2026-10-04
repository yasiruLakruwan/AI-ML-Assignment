import cv2
import os
import tempfile


class VideoReader:

    def __init__(self, video_path: str):

        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        self.frame_count = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.duration = (
            self.frame_count / self.fps
            if self.fps > 0
            else 0
        )

    # =========================================================
    # VIDEO METADATA
    # =========================================================

    def metadata(self):

        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration
        }

    # =========================================================
    # SEQUENTIAL FRAME READING
    # =========================================================

    def frames(self, sample_fps=2):

        interval = 1 / sample_fps

        next_sample_time = 0

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            frame_number = int(
                self.cap.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )

            timestamp = frame_number / self.fps

            if timestamp >= next_sample_time:

                yield timestamp, frame

                next_sample_time += interval

        self.cap.release()

    # =========================================================
    # GET A FRAME AT A SPECIFIC TIME
    # =========================================================

    def get_frame_at_time(self, timestamp):

        if timestamp < 0:
            return None

        if timestamp > self.duration:
            return None

        # Use a separate capture object so that
        # temporal access does not disturb the main
        # sequential video reader.

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            return None

        frame_number = int(
            timestamp * self.fps
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        cap.release()

        if not success:
            return None

        return frame

    # =========================================================
    # GET TEMPORAL CONTEXT
    # =========================================================

    def get_temporal_context(
        self,
        current_timestamp,
        previous_seconds=8,
        following_seconds=5,
        sample_count=4
    ):

        previous_paths = self._get_segment(
            start_time=max(
                0,
                current_timestamp - previous_seconds
            ),
            end_time=current_timestamp,
            sample_count=sample_count,
            prefix="previous"
        )

        following_paths = self._get_segment(
            start_time=current_timestamp,
            end_time=min(
                self.duration,
                current_timestamp + following_seconds
            ),
            sample_count=sample_count,
            prefix="following"
        )

        return {
            "previous": previous_paths,
            "following": following_paths
        }

    # =========================================================
    # GET FRAMES FROM A TIME SEGMENT
    # =========================================================

    def _get_segment(
        self,
        start_time,
        end_time,
        sample_count,
        prefix
    ):

        if start_time >= end_time:
            return []

        # -----------------------------------------
        # Create temporary directory
        # -----------------------------------------

        temp_dir = tempfile.mkdtemp(
            prefix="elder_temporal_"
        )

        # -----------------------------------------
        # Calculate timestamps
        # -----------------------------------------

        if sample_count == 1:

            timestamps = [
                (start_time + end_time) / 2
            ]

        else:

            step = (
                end_time - start_time
            ) / (sample_count - 1)

            timestamps = [
                start_time + (i * step)
                for i in range(sample_count)
            ]

        frame_paths = []

        # -----------------------------------------
        # Extract frames
        # -----------------------------------------

        for index, timestamp in enumerate(
            timestamps
        ):

            frame = self.get_frame_at_time(
                timestamp
            )

            if frame is None:
                continue

            filename = (
                f"{prefix}_{index}.jpg"
            )

            image_path = os.path.join(
                temp_dir,
                filename
            )

            success = cv2.imwrite(
                image_path,
                frame
            )

            if success:

                frame_paths.append(
                    image_path
                )

        return frame_paths
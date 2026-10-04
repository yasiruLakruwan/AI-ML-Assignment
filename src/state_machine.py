class StateMachine:

    def __init__(self):

        self.current_state = None
        self.previous_state = None

        self.state_start_time = None

        self.timeline = []

        self.pending_state = None
        self.pending_start = None

    def update(
        self,
        predicted_state,
        timestamp
    ):

        # First state
        if self.current_state is None:

            self.current_state = predicted_state
            self.state_start_time = timestamp

            return None

        # Same state
        if predicted_state == self.current_state:

            return None

        # New candidate state
        if self.pending_state != predicted_state:

            self.pending_state = predicted_state
            self.pending_start = timestamp

            return None

        # Candidate state has remained stable
        stable_duration = (
            timestamp - self.pending_start
        )

        if stable_duration >= 1.5:

            segment = {
                "start": self.state_start_time,
                "end": self.pending_start,
                "state": self.current_state
            }

            self.timeline.append(segment)

            self.previous_state = self.current_state

            self.current_state = self.pending_state

            self.state_start_time = self.pending_start

            self.pending_state = None
            self.pending_start = None

            return {
                "type": "STATE_CHANGE",
                "previous_state": self.previous_state,
                "current_state": self.current_state,
                "timestamp": timestamp
            }

        return None

    def finalize(self, final_timestamp):

        if self.current_state is None:
            return

        self.timeline.append(
            {
                "start": self.state_start_time,
                "end": final_timestamp,
                "state": self.current_state
            }
        )
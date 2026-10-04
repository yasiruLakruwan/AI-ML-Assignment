class BedEventDetector:

    def __init__(self):

        self.pending_exit = False
        self.pending_return = False

        self.bed_exit_count = 0
        self.bed_return_count = 0

        self.events = []

    def check_exit(
        self,
        previous_state,
        current_state,
        timestamp
    ):

        if previous_state in [
            "LYING_IN_BED",
            "SITTING_ON_BED"
        ] and current_state == "STANDING":

            self.pending_exit = True

        elif (
            self.pending_exit
            and current_state == "WALKING"
        ):

            self.bed_exit_count += 1

            event = {
                "event": "BED_EXIT",
                "confirmed_time": timestamp,
                "previous_state": previous_state,
                "current_state": current_state,
                "confidence": 0.85,
                "decision": "MONITOR"
            }

            self.events.append(event)

            self.pending_exit = False

            return event

        return None

    def check_return(
        self,
        previous_state,
        current_state,
        timestamp
    ):

        if current_state == "SITTING_ON_BED":

            self.pending_return = True

        elif (
            self.pending_return
            and current_state == "LYING_IN_BED"
        ):

            self.bed_return_count += 1

            event = {
                "event": "RETURN_TO_BED",
                "confirmed_time": timestamp,
                "previous_state": previous_state,
                "current_state": current_state,
                "confidence": 0.90,
                "decision": "NORMAL"
            }

            self.events.append(event)

            self.pending_return = False

            return event

        return None
from collections import defaultdict


class ReportGenerator:

    def duration_summary(self, timeline):

        totals = defaultdict(float)

        for segment in timeline:

            duration = (
                segment["end"]
                - segment["start"]
            )

            totals[
                segment["state"].lower()
            ] += duration

        return dict(totals)
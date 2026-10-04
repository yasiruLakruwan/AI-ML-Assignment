import csv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class Evaluator:

    def __init__(self):
        self.results = []

    def add_result(
        self,
        timestamp,
        prediction
    ):

        self.results.append({
            "timestamp": timestamp,
            "prediction": prediction
        })

    def save_predictions(
        self,
        filepath
    ):

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "prediction"
                ]
            )

            writer.writeheader()

            writer.writerows(
                self.results
            )

    @staticmethod
    def calculate_metrics(
        ground_truth,
        predictions
    ):

        precision = precision_score(
            ground_truth,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            ground_truth,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            ground_truth,
            predictions,
            average="weighted",
            zero_division=0
        )

        accuracy = accuracy_score(
            ground_truth,
            predictions
        )

        print("\n==============================")
        print("SYSTEM EVALUATION")
        print("==============================")

        print(
            f"Accuracy  : {accuracy:.2%}"
        )

        print(
            f"Precision : {precision:.2%}"
        )

        print(
            f"Recall    : {recall:.2%}"
        )

        print(
            f"F1-score  : {f1:.2%}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                ground_truth,
                predictions,
                zero_division=0
            )
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                ground_truth,
                predictions
            )
        )
import csv
import os


CSV_FILE = "experiment_logs.csv"


def initialize_csv():

    if not os.path.exists(CSV_FILE):

        with open(CSV_FILE, mode="w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "attempt",
                "sample_size",
                "position",
                "most_common_byte",
                "bias_ratio",
                "confidence",
                "validated",
                "action"
            ])


def log_experiment(data):

    with open(CSV_FILE, mode="a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            data["attempt"],
            data["sample_size"],
            data["position"],
            data["most_common_byte"],
            round(data["bias_ratio"], 4),
            round(data["confidence"], 4),
            data["validated"],
            data["action"]
        ])
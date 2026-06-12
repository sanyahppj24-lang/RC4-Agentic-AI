import pandas as pd
from collections import Counter


def analyze_positions(dataset_path, max_position=10):

    df = pd.read_csv(dataset_path)

    results = []

    for position in range(max_position):

        column = f"byte_{position}"

        values = df[column].tolist()

        frequency = Counter(values)

        total_samples = len(values)

        most_common_byte, actual_frequency = (
            frequency.most_common(1)[0]
        )

        expected_probability = 1 / 256

        observed_probability = (
            actual_frequency / total_samples
        )

        bias_ratio = (
            observed_probability /
            expected_probability
        )

        target_probability = 0.0078

        distance_from_target = abs(
            observed_probability -
            target_probability
        )

        result = {
            "position": position,
            "most_common_byte": most_common_byte,
            "occurrences": actual_frequency,
            "observed_probability": observed_probability,
            "expected_probability": expected_probability,
            "bias_ratio": bias_ratio,
            "distance_from_target": distance_from_target
        }

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="bias_ratio",
        ascending=False
    )

    results_df.to_csv(
        "data/position_bias_analysis.csv",
        index=False
    )

    return results_df


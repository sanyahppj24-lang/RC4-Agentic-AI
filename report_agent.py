def generate_report(best_result, memory):

    report = "\n========== RC4 BIAS REPORT ==========\n"

    report += (
        f"\nMost biased byte position discovered: "
        f"{int(best_result['position'])}"
    )

    report += (
        f"\nMost common byte: "
        f"{int(best_result['most_common_byte'])}"
    )

    report += (
        f"\nObserved probability: "
        f"{round(best_result['observed_probability'], 6)}"
    )

    report += (
        f"\nExpected probability: "
        f"{round(best_result['expected_probability'], 6)}"
    )

    report += (
        f"\nBias ratio: "
        f"{round(best_result['bias_ratio'], 4)}"
    )

    report += (
        f"\nDistance from target: "
        f"{round(best_result['distance_from_target'], 6)}"
    )

    report += "\n\n========== MEMORY ==========\n"

    for experiment in memory:

        report += (
            f"\nAttempt {experiment['attempt']} | "
            f"Samples: {experiment['sample_size']} | "
            f"Position: {experiment['position']} | "
            f"Bias Ratio: {round(experiment['bias_ratio'], 4)}"
        )

    report += "\n\n========== END ==========\n"

    return report
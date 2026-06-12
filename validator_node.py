def validate_bias(best_result):

    observed_probability = (
        best_result["observed_probability"]
    )

    target_probability = 0.0078

    confidence = (
        observed_probability /
        target_probability
    )

    validated = (
        observed_probability >=
        target_probability
    )

    return {
        "validated": validated,
        "confidence": confidence
    }

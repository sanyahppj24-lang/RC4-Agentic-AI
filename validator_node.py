import math


def validate_bias(best_result):

    observed = best_result[
        "observed_probability"
    ]

    expected = best_result[
        "expected_probability"
    ]

    sample_size = best_result[
        "sample_size"
    ]

    # =====================================================
    # Z SCORE
    # =====================================================

    numerator = (
        observed - expected
    )

    denominator = math.sqrt(
        (
            expected
            * (1 - expected)
        )
        / sample_size
    )

    z_score = numerator / denominator

    # =====================================================
    # VALIDATION
    # =====================================================

    validated = z_score > 3

    return {

        "confidence":
            round(z_score, 4),

        "validated":
            validated
    }
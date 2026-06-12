import pandas as pd


def validate_dataset(path):

    df = pd.read_csv(path)

    issues = []

    # Check empty values
    if df.isnull().values.any():

        issues.append("Missing values detected")

        df = df.dropna()

    # Check duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:

        issues.append(f"{duplicates} duplicate rows removed")

        df = df.drop_duplicates()

    # Check byte ranges
    byte_columns = [
        col for col in df.columns
        if col.startswith("byte_")
    ]

    for col in byte_columns:

        invalid = df[
            (df[col] < 0) | (df[col] > 255)
        ]

        if len(invalid) > 0:

            issues.append(f"Invalid values in {col}")

            df = df[
                (df[col] >= 0) &
                (df[col] <= 255)
            ]

    # Save repaired dataset
    repaired_path = "data/repaired_rc4_dataset.csv"

    df.to_csv(repaired_path, index=False)

    print("\n[DATASET VALIDATOR]")

    if issues:

        print("Issues found:")

        for issue in issues:
            print("-", issue)

    else:

        print("Dataset valid.")

    print("Final rows:", len(df))

    return {
        "dataset_path": repaired_path,
        "issues": issues,
        "valid": len(issues) == 0
    }
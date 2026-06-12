import random
import string
import pandas as pd

from rc4_tool import generate_keystream


def generate_dataset(num_samples=50000, key_length=16, keystream_length=10):
    data = []

    for sample_id in range(num_samples):
        key = ''.join(
            random.choice(string.ascii_letters)
            for _ in range(key_length)
        )

        keystream = generate_keystream(
            key,
            length=keystream_length
        )

        row = {
            "sample_id": sample_id,
            "key": key
        }

        for i, byte in enumerate(keystream):
            row[f"byte_{i}"] = byte

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("data/rc4_dataset.csv", index=False)

    return "data/rc4_dataset.csv"
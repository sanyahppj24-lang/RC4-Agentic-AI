from rc4 import ksa, prga


def generate_keystream(key, length=10):
    S = ksa(key)
    stream = prga(S)

    keystream = []

    for _ in range(length):
        keystream.append(next(stream))

    return keystream
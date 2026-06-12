def ksa(key):
    key = [ord(c) for c in key]
    S = list(range(256))
    j = 0

    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]

    return S


def prga(S):
    i = 0
    j = 0

    while True:
        i = (i + 1) % 256
        j = (j + S[i]) % 256

        S[i], S[j] = S[j], S[i]

        K = S[(S[i] + S[j]) % 256]
        yield K


def encrypt(key, plaintext):
    S = ksa(key)
    keystream = prga(S)

    ciphertext = []

    for char in plaintext:
        cipher_byte = ord(char) ^ next(keystream)
        ciphertext.append(cipher_byte)

    return ciphertext


def decrypt(key, ciphertext):
    S = ksa(key)
    keystream = prga(S)

    plaintext = ""

    for byte in ciphertext:
        plain_char = chr(byte ^ next(keystream))
        plaintext += plain_char

    return plaintext
def extract_keystream(plaintext, ciphertext):
    keystream = []

    for p, c in zip(plaintext, ciphertext):
        keystream_byte = ord(p) ^ c
        keystream.append(keystream_byte)

    return keystream
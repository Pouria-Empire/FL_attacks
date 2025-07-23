# In crypto_utils.py
from Crypto.Cipher import AES
import pickle
import numpy as np
from typing import List

# WARNING: This is a hardcoded key for simulation ONLY.
SECRET_KEY = b'this_is_a_32_byte_secret_key_!!'

def encrypt_params(params: List[np.ndarray]) -> bytes:
    """Serializes and encrypts a list of NumPy arrays."""
    data_bytes = pickle.dumps(params)
    cipher = AES.new(SECRET_KEY, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data_bytes)
    # Package nonce, tag, and ciphertext together for transport
    return pickle.dumps({"nonce": cipher.nonce, "ciphertext": ciphertext, "tag": tag})

def decrypt_params(encrypted_bytes: bytes) -> List[np.ndarray]:
    """Decrypts and deserializes a list of NumPy arrays."""
    encrypted_dict = pickle.loads(encrypted_bytes)
    nonce = encrypted_dict["nonce"]
    tag = encrypted_dict["tag"]
    ciphertext = encrypted_dict["ciphertext"]
    cipher = AES.new(SECRET_KEY, AES.MODE_EAX, nonce=nonce)
    decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag)
    return pickle.loads(decrypted_bytes)
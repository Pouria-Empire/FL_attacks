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


def chaotic_map_obfuscate(params: List[np.ndarray], key: float = 3.99, seed: float = 0.5) -> List[np.ndarray]:
    """
    Obfuscates parameters using a deterministic chaotic logistic map.
    This is a lightweight perturbation, NOT secure encryption.
    """
    obfuscated_params = []
    x = seed # Initial seed for the map
    for p in params:
        # Generate a chaotic mask with the same shape as the parameter
        mask = np.zeros_like(p, dtype=np.float32)
        flat_mask = mask.flatten()
        for i in range(len(flat_mask)):
            x = key * x * (1 - x) # Logistic map equation
            flat_mask[i] = x
        mask = flat_mask.reshape(p.shape)
        
        # Apply the mask by adding it to the parameters
        obfuscated_params.append(p + (mask - 0.5)) # Center the mask around 0
        
    return obfuscated_params

def chaotic_map_deobfuscate(params: List[np.ndarray], key: float = 3.99, seed: float = 0.5) -> List[np.ndarray]:
    """Reverses the chaotic map obfuscation."""
    deobfuscated_params = []
    x = seed
    for p in params:
        mask = np.zeros_like(p, dtype=np.float32)
        flat_mask = mask.flatten()
        for i in range(len(flat_mask)):
            x = key * x * (1 - x)
            flat_mask[i] = x
        mask = flat_mask.reshape(p.shape)
        
        # Reverse the obfuscation by subtracting the mask
        deobfuscated_params.append(p - (mask - 0.5))
        
    return deobfuscated_params
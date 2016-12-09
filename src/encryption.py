from Crypto.Cipher import AES
from hashlib import sha256
from os import urandom
from encoding import encode, decode

###############################################################################
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARN#
#                                                                             #
# This implements NON-AUTHENTICATED encryption / decryption.                  #
#                                                                             #
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARN#
###############################################################################

def encrypt(model, key, plaintext):
    """Encrypts the plaintext using AES with a model-based transformation.

    Note: If the plaintext does not end with a boundary (e.g. space), it will
    be appended. The boundary is defined in `model.boundary`.

    Example:
        >> ciphertext = encrypt(model, "foo", "bar")
        >> decrypt(model, "foo", ciphertext)
        "BAR "

    Args:
        model (Keras Model): A model that has been trained on a domain related
            to the plaintext being encrypted.

        key (string): A key to use to encrypt the plaintext.

        plaintext (string): The plaintext to be encrypted. The plaintext
            should only contain values that are present in the `model`'s
            alpahbet.

    Returns (bytes):
        The encrypted ciphertext.

    Raises:
        ValueError: If `plaintext` contains an item that isn't in the `model`'s
            alphabet.

        Exception: If padding the encoded plaintext fails. This is a
            non-deterministic process. The probability of this happening is
            highly unlikely, but not impossible. If your model has a boundary
            that occurs with a low-probability and you're getting this
            exception, increase your model's max_padding_trials attribute.
    """
    iv = urandom(AES.block_size)

    encoded = encode(model, plaintext, AES.block_size)
    encrypted = _get_cipher(key, iv).encrypt(encoded)

    return iv + encrypted

def decrypt(model, key, ciphertext):
    """Decrypts the ciphertext using AES with a model-based transformation.

    Example:
        >> ciphertext = encrypt(model, "foo", "bar")
        >> decrypt(model, "foo", ciphertext)
        "BAR "

    Args:
        model (Keras Model): The model that was used when encrypting the
            provided ciphertext.

        key (string): A key to use to decrypt the ciphertext.

        ciphertext (bytes): The ciphertext to be decrypted.

    Returns (string):
        The decrypted plaintext.
    """
    iv = ciphertext[:AES.block_size]
    ciphertext = ciphertext[AES.block_size:]

    decrypted = _get_cipher(key, iv).decrypt(ciphertext)
    decoded = decode(model, decrypted)

    return decoded

def _get_cipher(key, iv):
    """ Returns an AES cipher in CFB mode. """
    return AES.new(_transform_key(key), AES.MODE_CFB, iv)

def _transform_key(key):
    """ Securely hashes a key to a 32 byte block. """
    return sha256(bytes(key, 'utf-8')).digest()

from model import load_model
from encryption import encrypt, decrypt

PASSWORD = "PASSWORD"
TEST_INPUT = "Hello, this is a test. A mega mega mega"
TEST_INPUT = "attack at dawn on the eastern front with 1,000 men."

def test():
    model = load_model("../models/mil_512")
    enc = encrypt(model, PASSWORD, transform(TEST_INPUT))

    words = [word.strip().upper() for word in open("/usr/share/dict/words").readlines()]
    words = set(word for word in words if len(word) == 4)
    for word in words:
        print("%s: %s"%(word, ''.join(decrypt(model, word, enc))))

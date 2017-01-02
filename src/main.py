import argparse
from sys import stdout, argv, exit
from getpass import getpass
from base64 import b64encode, b64decode
from model import load_model
from encryption import encrypt, decrypt
from util.io import confirmed_get_pass, read_file

def encrypt_command(args):
    key = args.key
    if key == None:
        key = confirmed_get_pass("Encryption Key: ", "Confirm Encryption Key: ")
        if key == None:
            print("Keys didn't match. Exiting.")
            exit(2)

    model = load_model(args.config)
    # NOTE We rstrip() the plaintext. Input tends to end in newlines and it can
    # be a signal to an attacker (e.g. by checking if the decoy output has a newline).
    plaintext = read_file(args.file).rstrip()
    encrypted = encrypt(model, key, plaintext)
    encoded = str(b64encode(encrypted), 'utf-8')
    stdout.write(encoded)

def decrypt_command(args):
    key = args.key
    if key == None:
        key = getpass("Decryption Key: ")

    model = load_model(args.config)
    encoded = read_file(args.file)
    ciphertext = b64decode(encoded)
    decrypted = decrypt(model, key, ciphertext)
    stdout.write(decrypted)

def train_command(args):
    model = load_model(args.config)
    data = read_file(args.data)
    model.train(data)

def sample_command(args):
    model = load_model(args.config)
    size = int(args.size)
    print(model.sample(size))

def main():
    parser = argparse.ArgumentParser(
            prog='menc',
            formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="""\
Prototype of a modeled encryption implementation.

Example Usage:

  Encryption / Descryption
  =============================================================================

  - Encrypt a file:
    $ menc encrypt -c models/mil_512/config.json -f filename

  - Encrypt from stdin:
    $ echo 'Hello World!' | menc encrypt -c models/mil_512/config.json

  - Encrypt from stdin, key provided as arg:
    $ echo 'Hello World!' | menc encrypt -c models/mil_512/config.json -k foo

  - Store encrypted result into a file:
    $ echo 'Hello World!' | menc encrypt -c models/mil_512/config.json > encrypted_file

  - Decrypt a file:
    $ menc decrypt -c models/mil_512/config.json -f filename

  - Decrypt from stdin:
    $ cat encrypted | menc decrypt -c models/mil_512/config.json

  - Decrypt from stdin, key provided as arg:
    $ cat encrypted | menc decrypt -c models/mil_512/config.json -k foo

  - Store decrypted result into a file:
    $ cat encrypted | menc -c models/mil_512/config.json > decrypted_file

  - Round-trip (encrypt and then decrypt):
    $ echo 'Hello world!' | menc encrypt -c models/mil_512/config.json -k foo | menc decrypt -c models/mil_512/config.json -k foo

  Training
  =============================================================================

  - Train from data in a file:
    $ menc train -c models/mil_512/config.json -d models/mil_512/data.txt

  - Train from stdin:
    $ cat models/mil_512/data.txt | menc train -c models/mil_512/config.json

  Sampling
  =============================================================================

  - Generate a random sequence of length 100:
    $ menc sample -c models/mil_512/config.json -s 100""")
    subparsers = parser.add_subparsers()

    encrypt_parser = subparsers.add_parser('encrypt', help="Encrypt a plaintext.")
    encrypt_parser.add_argument('-c', '--config', metavar="CONFIG_PATH", help="Path to the model config.", required=True)
    encrypt_parser.add_argument('-k', '--key', help="The string to use as the encryption key. If ommitted, a password prompt will securely ask for one. Note: Providing a key on the command-line may store the key in your shell history.")
    encrypt_parser.add_argument('-f', '--file', help="File to encrypt. Reads stdin if not provided.")
    encrypt_parser.set_defaults(func=encrypt_command)

    decrypt_parser = subparsers.add_parser('decrypt', help="Decrypt a ciphertext.")
    decrypt_parser.add_argument('-c', '--config', metavar="CONFIG_PATH", help="Path to the model config.", required=True)
    decrypt_parser.add_argument('-k', '--key', help="The string to use as the decryption key. If ommitted, a password prompt will securely ask for one. Note: Providing a key on the command-line may store the key in your shell history.")
    decrypt_parser.add_argument('-f', '--file', help="File to decrypt. Reads stdin if not provided.")
    decrypt_parser.set_defaults(func=decrypt_command)

    train_parser = subparsers.add_parser('train', help="Train a model on a given set of data.")
    train_parser.add_argument('-c', '--config', metavar="CONFIG_PATH", help="Path to the model config.", required=True)
    train_parser.add_argument('-d', '--data', help="Path to data to train on.")
    train_parser.set_defaults(func=train_command)

    sample_parser = subparsers.add_parser('sample', help="Sample the model by generating a random sequence from it.")
    sample_parser.add_argument('-c', '--config', metavar="CONFIG_PATH", help="Path to the model config.", required=True)
    sample_parser.add_argument('-s', '--size', help="Length of the sequence to generate.")
    sample_parser.set_defaults(func=sample_command)

    args = parser.parse_args()

    if 'func' not in args:
        parser.print_help()
        exit(1)

    args.func(args)

if __name__ == "__main__":
    main()

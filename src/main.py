import argparse
from sys import stdout, argv, exit
from getpass import getpass
from base64 import a85encode, a85decode
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

    model = load_model(args.model)
    plaintext = read_file(args.file)
    encrypted = encrypt(model, key, plaintext)
    encoded = str(a85encode(encrypted), 'utf-8')
    stdout.write(encoded)

def decrypt_command(args):
    key = args.key
    if key == None:
        key = getpass("Decryption Key: ")

    model = load_model(args.model)
    encoded = read_file(args.file)
    ciphertext = a85decode(encoded)
    decrypted = decrypt(model, key, ciphertext)
    stdout.write(decrypted)

def main():
    parser = argparse.ArgumentParser(
            prog='menc',
            formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="""\
Prototype of a modeled encryption implementation.

Example Usage:

  - Encrypt a file:
    $ menc encrypt -m models/mil_512 -f filename

  - Encrypt from stdin:
    $ echo "Hello World!" | menc encrypt -m models/mil_512

  - Encrypt from stdin, key provided as arg:
    $ echo "Hello World!" | menc encrypt -m models/mil_512 -k foo

  - Store encrypted result into a file:
    $ echo "Hello World!" | menc encrypt -m models/mil_512 > encrypted_file

  - Decrypt a file:
    $ menc decrypt -m models/mil_512 -f filename

  - Decrypt from stdin:
    $ cat encrypted | menc decrypt -m models/mil_512

  - Decrypt from stdin, key provided as arg:
    $ cat encrypted | menc decrypt -m models/mil_512 -k foo

  - Store decrypted result into a file:
    $ cat encrypted | menc -m models/mil_512 > decrypted_file

  - Round-trip (encrypt and then decrypt):
    $ echo 'Hello world!" | menc encrypt -m models/mil_512 -k foo | menc decrypt -m models/mil_512 -k foo""")
    subparsers = parser.add_subparsers()

    encrypt_parser = subparsers.add_parser('encrypt', help="Encrypt a plaintext.")
    encrypt_parser.add_argument('-m', '--model', metavar="MODEL_PATH", help="Path to the model directory.", required=True)
    encrypt_parser.add_argument('-k', '--key', help="The string to use as the encryption key. If ommitted, a password prompt will securely ask for one. Note: Providing a key on the command-line may store the key in your shell history.")
    encrypt_parser.add_argument('-f', '--file', help="File to encrypt. Reads stdin if not provided.")
    encrypt_parser.set_defaults(func=encrypt_command)

    decrypt_parser = subparsers.add_parser('decrypt', help="Decrypt a ciphertext.")
    decrypt_parser.add_argument('-m', '--model', metavar="MODEL_PATH", help="Path to the model directory.", required=True)
    decrypt_parser.add_argument('-k', '--key', help="The string to use as the decryption key. If ommitted, a password prompt will securely ask for one. Note: Providing a key on the command-line may store the key in your shell history.")
    decrypt_parser.add_argument('-f', '--file', help="File to decrypt. Reads stdin if not provided.")
    decrypt_parser.set_defaults(func=decrypt_command)

    args = parser.parse_args()

    if 'func' not in args:
        parser.print_help()
        exit(1)

    args.func(args)

if __name__ == "__main__":
    main()

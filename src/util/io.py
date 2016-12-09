from sys import stdout, stdin
from getpass import getpass

def confirmed_get_pass(message, confirmation_message):
    """Securely gets input from the console twice and confirms that they match.

    Args:
        message (string): The message to show to the user for the first request.

        confirmation_message (string:The message to show to the user for the
            second request.

    Returns:
        None if the inputs don't match, otherwise returns the input.
    """
    input1 = getpass(message)
    input2 = getpass(confirmation_message)

    if input1 != input2:
        return None

    return input1

def read_file(filename):
    """Reads the contents of a file. If the filename is None or '-', defaults
    to stdin.

    Args:
        filename (string): The filename to read.

    Returns:
        The contents of the file or stdin.
    """
    if filename == '-' or filename == None:
        return stdin.read()
    else:
        with open(filename) as fin:
            return fin.read()

""" Keras writes to stderr unconditionally, so we disable stderr while importing. """
import sys
from os import devnull

#################################
# Disable stderr
#################################
_old_stderr = sys.stderr
sys.stderr = open(devnull, 'w')
#################################

from keras.models import model_from_json

#################################
# Enable stderr
#################################
sys.stderr = _old_stderr
#################################

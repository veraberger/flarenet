# Standard library
import os  # noqa
import logging
from .version import __version__

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger("flarenet")


from .tessprep import TessStar, get_TESS_data  # noqa
from .flarenet import Flarenet

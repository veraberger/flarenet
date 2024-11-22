# Standard library
import os  # noqa
import logging
from .version import __version__

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger("flarenet")

from .postprocessing import *  # noqa
from .cosmic_ray_extension import *
from .execute_postprocessing import *
from .flarenet_model import flarenet

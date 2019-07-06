from pkg_resources import get_distribution, DistributionNotFound

from ptychotomo.objects import *
from ptychotomo.ptychofft import *
from ptychotomo.radonusfft import *
from ptychotomo.solver import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from pkg_resources import get_distribution, DistributionNotFound

from libtike.cufft.ptycho import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

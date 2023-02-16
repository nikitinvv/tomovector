from pkg_resources import get_distribution, DistributionNotFound

from tomovector.fourier_rec import *
from tomovector.line_rec import *
from tomovector.solver_tomo import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
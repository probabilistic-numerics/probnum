"""
Import convenience functions in optim.py to create an
intuitive, numpy-like interface.

Note
----
Local import, because with a global import this does not seem
to work.
"""

from .ivptofilter import *
from .odefiltsmooth import *
from .prior import *

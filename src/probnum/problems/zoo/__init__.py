"""Collection of test problems for probabilistic numerical methods."""
import os
import sys
from pathlib import Path

# Directory containing data of the problem zoo
PROBLEMZOO_DIR = "probnum/problems/zoo/"
if sys.platform != "win32":
    PROBLEMZOO_DIR = "." + PROBLEMZOO_DIR

PROBLEMZOO_DIR = os.path.join(Path.home(), PROBLEMZOO_DIR)

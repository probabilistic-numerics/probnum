"""Utility classes and functions for interfacing the SuiteSparse Matrix Collection.

This implementation is based on Sudarshan Raghunathan's SSGETPY package
(https://github.com/drdarshan/ssgetpy).
"""

import os
import sys
from pathlib import Path

# Directory containing data of the problem zoo
PROBLEMZOO_DIR = "probnum/problems/zoo/"
if sys.platform != "win32":
    PROBLEMZOO_DIR = "." + PROBLEMZOO_DIR

PROBLEMZOO_DIR = os.path.join(Path.home(), PROBLEMZOO_DIR)


# URLs and file paths to data
SUITESPARSE_ROOT_URL = "https://sparse.tamu.edu"
SUITESPARSE_INDEX_URL = os.path.join(SUITESPARSE_ROOT_URL, "files", "ssstats.csv")
SUITESPARSE_DIR = os.path.join(PROBLEMZOO_DIR, "linalg")
SUITESPARSE_DB = os.path.join(SUITESPARSE_DIR, "index.db")

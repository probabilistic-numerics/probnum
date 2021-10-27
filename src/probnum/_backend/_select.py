import json
import os
import pathlib

from . import Backend

BACKEND_FILE = pathlib.Path.home() / ".probnum.json"
BACKEND_FILE_KEY = "backend"

BACKEND_ENV_VAR = "PROBNUM_BACKEND"


def select_backend() -> Backend:
    if BACKEND_FILE.exists() and BACKEND_FILE.is_file():
        try:
            with BACKEND_FILE.open("r") as f:
                config = json.load(f)

            return Backend[config[BACKEND_FILE_KEY].upper()]
        except Exception:
            pass

    if BACKEND_ENV_VAR in os.environ:
        backend_str = os.environ[BACKEND_ENV_VAR].upper()

        if backend_str not in Backend:
            raise ValueError("TODO")

        return Backend[backend_str]

    return _select_via_import()


def _select_via_import() -> Backend:
    try:
        import jax  # pylint: disable=unused-import,import-outside-toplevel

        return Backend.JAX
    except ImportError:
        pass

    try:
        import torch  # pylint: disable=unused-import,import-outside-toplevel

        return Backend.PYTORCH
    except ImportError:
        pass

    return Backend.NUMPY

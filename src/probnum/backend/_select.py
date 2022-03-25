import enum
import json
import os
import pathlib


@enum.unique
class Backend(enum.Enum):
    JAX = enum.auto()
    TORCH = enum.auto()
    NUMPY = enum.auto()


BACKEND_FILE = pathlib.Path.home() / ".probnum.json"
BACKEND_FILE_KEY = "backend"

BACKEND_ENV_VAR = "PROBNUM_BACKEND"


def select_backend() -> Backend:
    backend_str = None

    if BACKEND_ENV_VAR in os.environ:
        backend_str = os.environ[BACKEND_ENV_VAR].upper()
    elif BACKEND_FILE.exists() and BACKEND_FILE.is_file():
        with BACKEND_FILE.open("r") as f:
            config = json.load(f)

        if BACKEND_FILE_KEY in config:
            backend_str = config[BACKEND_FILE_KEY].upper()

    if backend_str is not None:
        try:
            return Backend[backend_str]
        except KeyError as e:
            # TODO
            raise e from e

    return Backend.NUMPY


def _select_via_import() -> Backend:
    try:
        import jax  # pylint: disable=unused-import,import-outside-toplevel

        return Backend.JAX
    except ImportError:
        pass

    try:
        import torch  # pylint: disable=unused-import,import-outside-toplevel

        return Backend.TORCH
    except ImportError:
        pass

    return Backend.NUMPY

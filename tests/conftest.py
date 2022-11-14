from probnum import backend

import pytest


def pytest_configure(config: "_pytest.config.Config"):
    config.addinivalue_line(
        "markers", "skipif_backend(backend): Skip test for the given backend."
    )


def pytest_runtest_setup(item: pytest.Item):
    # Setup conditional backend skip
    skipped_backends = [
        mark.args[0] for mark in item.iter_markers(name="skipif_backend")
    ]

    if skipped_backends:
        if backend.BACKEND in skipped_backends:
            pytest.skip(f"Test skipped for backend {backend.BACKEND}.")

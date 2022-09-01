import pytest


@pytest.fixture(scope="module")
def seed() -> int:
    return 234

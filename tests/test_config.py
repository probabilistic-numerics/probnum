import pytest

import probnum


def test_register():
    probnum.config.register("dummy", 3.14, None)
    assert probnum.config.dummy == 3.14

    with pytest.raises(KeyError):
        probnum.config.register("dummy", 4.2, None)

    with probnum.config(dummy=9.9):
        assert probnum.config.dummy == 9.9

    probnum.config.dummy = 4.5
    assert probnum.config.dummy == 4.5

    with pytest.raises(KeyError):
        with probnum.config(unknown_config=False):
            pass

    with pytest.raises(KeyError):
        probnum.config.unknown_config = False

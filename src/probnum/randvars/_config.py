from probnum import config


def _register_default_config():
    _defaults = [
        ("covariance_inversion_damping", 1e-12),
    ]

    for k, v in _defaults:
        config.register(k, v)

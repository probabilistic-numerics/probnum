try:
    import jax
    from jax.numpy import all, any  # pylint: disable=redefined-builtin, unused-import

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:
    pass

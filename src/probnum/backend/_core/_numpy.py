from numpy import all, any  # pylint: disable=redefined-builtin, unused-import


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f

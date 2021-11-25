from probnum import backend, linops


def cast(a, dtype=None, casting="unsafe", copy=None):
    if isinstance(a, linops.LinearOperator):
        return a.astype(dtype=dtype, casting=casting, copy=copy)

    return backend.cast(a, dtype=dtype, casting=casting, copy=copy)

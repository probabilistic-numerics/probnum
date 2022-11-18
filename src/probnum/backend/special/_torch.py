"""Special functions in PyTorch."""

try:
    from torch.special import ndtr, ndtri
except ModuleNotFoundError:
    pass


def gamma(*args, **kwargs):
    raise NotImplementedError()


def kv(*args, **kwargs):
    raise NotImplementedError()

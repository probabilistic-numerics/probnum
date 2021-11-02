import torch


def gamma(x):
    return torch.exp(torch.special.lgamma(x))


def kv(*args, **kwargs):
    raise NotImplementedError()

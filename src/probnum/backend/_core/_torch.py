try:
    import torch

    torch.set_default_dtype(torch.double)
except ModuleNotFoundError:
    pass

import pytest

import probnum as pn
from probnum import backend

torch = pytest.importorskip("torch")


def test_hyperopt():
    lengthscale = torch.full((), 3.0)
    lengthscale.requires_grad_(True)

    def loss_fn():
        gp = pn.randprocs.GaussianProcess(
            mean=lambda x: backend.zeros_like(x, shape=x.shape[:-1]),
            cov=pn.kernels.ExpQuad(input_dim=1, lengthscale=lengthscale ** 2),
        )

        xs = backend.linspace(-1.0, 1.0, 10)
        ys = backend.sin(backend.pi * xs)

        fX = gp(xs[:, None])

        e = pn.randvars.Normal(mean=backend.zeros(10), cov=backend.eye(10))

        return -(fX + e).logpdf(ys)

    optimizer = torch.optim.LBFGS(params=[lengthscale], line_search_fn="strong_wolfe")

    before = loss_fn()

    for iter_idx in range(5):

        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        optimizer.step(closure)

    after = loss_fn()

    assert before >= after

    print()


if __name__ == "__main__":
    test_hyperopt()

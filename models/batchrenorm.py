import torch


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-4,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.num_features = num_features
        self.sample_noise = False

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return 10

    @property
    def dmax(self) -> torch.Tensor:
        return 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = x.std(dims, unbiased=False) + self.eps
            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            )
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            )
            if self.sample_noise is True:
                noise_mean = torch.normal(mean=self.sample_mean.fill_(1), std=self.sample_std_mean.detach())
                noise_var = torch.normal(mean=self.sample_mean.fill_(1), std=self.sample_std_var.detach())
                r *= noise_var.detach()
                d *= noise_mean.detach()
                r = r.detach()
                d = d.detach()
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")

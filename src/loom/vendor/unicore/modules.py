import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._weight_bf16: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "RMSNorm":
        self._weight_bf16 = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self._weight_bf16 = None
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            dtype = x.dtype
            if dtype == torch.bfloat16 and not self.training and not torch.is_grad_enabled():
                if self._weight_bf16 is None or self._weight_bf16.device != self.weight.device:
                    self._weight_bf16 = self.weight.detach().to(torch.bfloat16)
                w = self._weight_bf16
            else:
                self._weight_bf16 = None
                w = self.weight if self.weight.dtype == dtype else self.weight.to(dtype)
            return F.rms_norm(x, (x.shape[-1],), w, self.eps)
        dtype = x.dtype
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


class LayerNorm(nn.LayerNorm):
    pass

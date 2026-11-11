import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Pure-PyTorch drop-in replacement for torch_scatter.segment_csr."""
    num_segments = indptr.numel() - 1
    lengths = indptr[1:] - indptr[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(num_segments, device=src.device, dtype=torch.long),
        lengths,
    )
    out_shape = (num_segments,) + src.shape[1:]
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.index_add_(0, segment_ids, src)
    if reduce == "mean":
        denom = lengths.clamp(min=1).to(src.dtype)
        view_shape = (num_segments,) + (1,) * (src.ndim - 1)
        out = out / denom.view(view_shape)
    elif reduce != "sum":
        raise NotImplementedError(f"Unsupported reduce={reduce}")
    return out

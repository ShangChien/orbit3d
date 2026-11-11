"""Per-candidate counter streams, independent of batching and row removal."""

import hashlib
import torch


def candidate_seed(run_seed, candidate_id, retry=0):
    key = f"orbit3d:{run_seed}:{candidate_id}:{retry}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


class RowRandomStreams:
    def __init__(self, seeds):
        self.seeds = list(seeds)
        self.counters = [0] * len(seeds)

    def draw(self, rows, device):
        values = []
        for row in rows:
            key = f"{self.seeds[row]}:{self.counters[row]}".encode()
            bits = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
            values.append(bits / 2**53)
            self.counters[row] += 1
        return torch.tensor(values, dtype=torch.float64, device=device)


def inverse_cdf(probs, uniforms):
    mass = probs.double().sum(-1, keepdim=True)
    if not torch.isfinite(mass).all() or (mass <= 0).any() or (probs < 0).any():
        raise ValueError("Empty or invalid sampling support")
    cdf = (probs.double() / mass).cumsum(-1)
    cdf[:, -1] = 1.0
    return torch.searchsorted(cdf.contiguous(), uniforms[:, None].contiguous(), right=True).squeeze(-1)

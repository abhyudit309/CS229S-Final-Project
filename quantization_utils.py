import torch

def quantization(x: torch.Tensor, s: float, z: int, a_q: int, b_q: int) -> torch.Tensor:
    return torch.clip(torch.round(x / s + z, decimals=0), min=a_q, max=b_q).to(torch.int8)


def dequantization(x_q: torch.Tensor, s: float, z: int) -> torch.Tensor:
    x_q = x_q.to(torch.int32)
    x = s * (x_q - z)
    return x.to(torch.float32)


def get_quantization_params(a, b, a_q, b_q):
    s = (b - a) / (b_q - a_q)
    z = int((b * a_q - a * b_q) / (b - a))
    return s, z


def get_symmetric_range(x: torch.Tensor):
    beta = torch.max(x.max(), abs(x.min())).item()
    return -beta, beta


def get_affine_range(x: torch.Tensor):
    return x.min().item(), x.max().item()
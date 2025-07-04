import torch
import torch.nn.functional as F


def mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the mean absolute error over HxW for each [N, C]"""
    if torch.is_complex(x):
        x = x.abs()
    if torch.is_complex(y):
        y = y.abs()
    return torch.mean(torch.abs(x - y), dim=(-2, -1))  # mean over H, W


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error over HxW for each [N, C]"""
    if torch.is_complex(x):
        x = x.abs()
    if torch.is_complex(y):
        y = y.abs()
    return torch.mean(torch.square(x - y), dim=(-2, -1))


def dot_product(x: torch.Tensor, y: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Return complex dot product over HxW for each [N, C]"""
    prod = torch.sum(x * torch.conj(y), dim=(-2, -1))  # shape [N, C]
    if normalized:
        norm = torch.sum(x.abs() * y.abs(), dim=(-2, -1))
        return prod / norm
    else:
        return prod
    

def quality(x: torch.Tensor, y: torch.Tensor, squared: bool = True, inversed: bool = False) -> torch.Tensor:
    """Return the magnitude of the normalized dot product over HxW for each [N, C]"""
    q = dot_product(x, y, normalized=True).abs()  # shape [N, C]
    if squared:
        q = torch.square(q)
    if inversed:
        q = 1 - q
    return q


def pearson(x: torch.Tensor, y: torch.Tensor, inversed: bool = False) -> torch.Tensor:
    """Pearson correlation over HxW for each [N, C]"""
    if torch.is_complex(x):
        x = x.abs()
    if torch.is_complex(y):
        y = y.abs()

    x_mean = x.mean(dim=(-2, -1), keepdim=True)
    y_mean = y.mean(dim=(-2, -1), keepdim=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    cov = (x_centered * y_centered).mean(dim=(-2, -1))
    std_x = x.std(dim=(-2, -1))
    std_y = y.std(dim=(-2, -1))

    p = cov / (std_x * std_y + 1e-8)  # avoid divide-by-zero
    if inversed:
        p = 1 - p
    return p

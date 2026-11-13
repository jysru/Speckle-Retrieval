"""Spatial metrics returning one value per batch item and channel.

Masks broadcast over H,W without flattening batch dimensions. Undefined
correlations/overlaps (constant images or zero energy) are defined as zero.
"""
import torch


def _magnitude(x):
    return x.abs() if torch.is_complex(x) else x


def _masked(x, mask):
    return x if mask is None else torch.where(mask, x, 0)


def _mean(x, mask=None):
    if mask is None:
        return x.mean(dim=(-2, -1))
    count = torch.broadcast_to(mask, x.shape).sum(dim=(-2, -1)).clamp_min(1)
    return _masked(x, mask).sum(dim=(-2, -1)) / count


def mae(x, y, mask=None):
    """Mean absolute amplitude error over H,W."""
    return _mean((_magnitude(x) - _magnitude(y)).abs(), mask)


def mse(x, y, mask=None):
    """Mean squared amplitude error over H,W."""
    return _mean((_magnitude(x) - _magnitude(y)).square(), mask)


def normalized_mse(x, y, mask=None):
    """Amplitude MSE / mean target energy; zero targets use unscaled MSE."""
    y = _magnitude(y)
    energy = _mean(y.square(), mask)
    return mse(x, y, mask) / torch.where(energy > 0, energy, 1)


def fourier_r_factor(x, y):
    """Scaled L1 Fourier amplitude residual per image (OSS stage ranking).

    gamma = sum(|x|*|y|)/sum(|x|**2) is a least-squares amplitude scale;
    R_F = sum(abs(|y|-gamma*|x|))/sum(|y|). For an all-zero target,
    return the unscaled amplitude L1 norm instead of an undefined 0/0.
    """
    x, y = x.abs(), y.abs()
    energy = x.square().sum(dim=(-2, -1))
    gamma = (x * y).sum(dim=(-2, -1)) / torch.where(energy > 0, energy, 1)
    residual = (y - gamma[..., None, None] * x).abs().sum(dim=(-2, -1))
    total = y.sum(dim=(-2, -1))
    return torch.where(total > 0, residual / torch.where(total > 0, total, 1),
                       x.sum(dim=(-2, -1)))


def dot_product(x, y, normalized=True, mask=None):
    """Complex inner product, optionally normalized by the two L2 norms."""
    prod = _masked(x * y.conj(), mask).sum(dim=(-2, -1))
    if not normalized:
        return prod
    ex = _masked(x.abs().square(), mask).sum(dim=(-2, -1))
    ey = _masked(y.abs().square(), mask).sum(dim=(-2, -1))
    tiny = torch.finfo(ex.dtype).tiny
    norm = ex.clamp_min(tiny).sqrt() * ey.clamp_min(tiny).sqrt()
    return torch.where((ex > 0) & (ey > 0), prod / norm.clamp_min(tiny), 0)


def quality(x, y, squared=True, inversed=False, mask=None):
    """L2-normalized field overlap, invariant to global phase and scale."""
    q = dot_product(x, y, mask=mask).abs().clamp(max=1)
    q = q.square() if squared else q
    return 1 - q if inversed else q


def weighted_phase_agreement(x, y, squared=True, inversed=False, mask=None):
    """Legacy quality: weighted phase agreement, insensitive to amplitude mismatch."""
    prod = dot_product(x, y, normalized=False, mask=mask)
    norm = _masked(x.abs() * y.abs(), mask).sum(dim=(-2, -1))
    q = prod.abs() / norm.clamp_min(torch.finfo(norm.dtype).tiny)
    q = q.clamp(max=1)
    q = q.square() if squared else q
    return 1 - q if inversed else q


def pearson(x, y, inversed=False, mask=None):
    """Population Pearson correlation; constant or empty images return zero."""
    x, y = _magnitude(x), _magnitude(y)
    xc = x - _mean(x, mask)[..., None, None]
    yc = y - _mean(y, mask)[..., None, None]
    cov = _mean(xc * yc, mask)
    vx, vy = _mean(xc.square(), mask), _mean(yc.square(), mask)
    tiny = torch.finfo(cov.dtype).tiny
    norm = vx.clamp_min(tiny).sqrt() * vy.clamp_min(tiny).sqrt()
    p = torch.where((vx > 0) & (vy > 0), cov / norm.clamp_min(tiny), 0).clamp(-1, 1)
    return 1 - p if inversed else p

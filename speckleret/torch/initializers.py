import torch
import torch.nn as nn
import torch.optim as optim

from . import transforms
from . import metrics

import warnings


def _default_ops(y_shape):
    """Centered unitary FFT and its adjoint; y_shape retained for compatibility."""
    return transforms.fourier_transform, transforms.inverse_fourier_transform


def flat_phases(tensor):
    """Zero-phase complex field, retaining float32/float64 precision."""
    magnitude = tensor.abs()
    return torch.complex(magnitude, torch.zeros_like(magnitude))


def random_phases(tensor, generator=None):
    """Uniform phases in [-pi,pi), with an optional device-matched generator."""
    magnitude = tensor.abs()
    phase = _random_phases_from_tensor(magnitude, generator)
    return torch.polar(magnitude, phase)


def _flat_phases_from_tensor(ref_tensor):
    return torch.zeros_like(ref_tensor.real)


def _random_phases_from_tensor(ref_tensor, generator=None):
    return 2 * torch.pi * torch.rand(ref_tensor.shape, dtype=ref_tensor.real.dtype,
                                   device=ref_tensor.device, generator=generator) - torch.pi


def _nchw(magnitude):
    if magnitude.ndim == 2:
        return magnitude[None, None]
    if magnitude.ndim != 4:
        raise ValueError('Expected HW or NCHW magnitudes')
    return magnitude


def thresholded_random_phase(measured_magnitude, mask=None, drop_fraction=0.25,
                             generator=None):
    """Experimental random-phase IFFT after discarding the weakest fraction.

    This is a heuristic, not spectral initialization. Returns NCHW fields.
    """
    if not 0 <= drop_fraction < 1:
        raise ValueError('drop_fraction must be in [0,1)')
    mag = _nchw(measured_magnitude.abs())
    threshold = torch.quantile(mag.flatten(-2), drop_fraction, dim=-1)[..., None, None]
    kept = torch.where(mag >= threshold, mag, 0)
    x = transforms.inverse_fourier_transform(random_phases(kept, generator=generator))
    if mask is not None:
        x = transforms.apply_support(x, mask)
    energy = x.abs().square().sum((-2, -1), keepdim=True).sqrt()
    target = mag.square().sum((-2, -1), keepdim=True).sqrt()
    return x * target / energy.clamp_min(torch.finfo(mag.dtype).eps)


def truncated_spectral(measured_magnitude, mask=None, n_iter=50, alpha=0.25,
                       generator=None):
    """Deprecated alias; n_iter is unused, alpha is the fraction discarded."""
    warnings.warn('truncated_spectral is a random-phase heuristic; use '
                  'thresholded_random_phase(drop_fraction=alpha)',
                  DeprecationWarning, stacklevel=2)
    return thresholded_random_phase(measured_magnitude, mask, alpha, generator)


def autocorrelation_svd(measured_magnitude, mask=None, generator=None):
    """Experimental rank-one autocorrelation SVD heuristic, not spectral PR.

    Each batch/channel is processed independently. No convergence guarantees.
    """
    mag = _nchw(measured_magnitude.abs())
    autocorr = transforms.inverse_fourier_transform(mag.square())
    u, _, _ = torch.linalg.svd(autocorr.real, full_matrices=False)
    amplitude = u[..., :, :1].expand_as(mag)
    x = random_phases(amplitude, generator=generator)
    if mask is not None:
        x = transforms.apply_support(x, mask)
    norm = x.abs().square().sum((-2, -1), keepdim=True).sqrt()
    target = mag.square().sum((-2, -1), keepdim=True).sqrt()
    return x * target / norm.clamp_min(torch.finfo(mag.dtype).eps)


def spectral(measured_magnitude, mask=None, generator=None):
    """Deprecated alias for the experimental autocorrelation_svd heuristic."""
    warnings.warn('spectral is an autocorrelation SVD heuristic; use '
                  'autocorrelation_svd explicitly', DeprecationWarning, stacklevel=2)
    return autocorrelation_svd(measured_magnitude, mask, generator)


def random_phase_multistart(measured_magnitude, mask=None, n_trials=5, generator=None):
    """Select random-phase candidates by amplitude residual per batch/channel.

    Without an object mask candidates all satisfy the Fourier magnitude, so
    ranking cannot identify the true phase. Returns NCHW fields.
    """
    if n_trials < 1:
        raise ValueError('n_trials must be positive')
    mag = _nchw(measured_magnitude.abs())
    best, best_loss = None, None
    for _ in range(n_trials):
        x = transforms.inverse_fourier_transform(random_phases(mag, generator=generator))
        if mask is not None:
            x = transforms.apply_support(x, mask)
        loss = metrics.mse(transforms.fourier_transform(x), mag)
        if best is None:
            best, best_loss = x, loss
        else:
            better = loss < best_loss
            best = torch.where(better[..., None, None], x, best)
            best_loss = torch.minimum(loss, best_loss)
    norm = best.abs().square().sum((-2, -1), keepdim=True).sqrt()
    return best * mag.square().sum((-2, -1), keepdim=True).sqrt() / norm.clamp_min(torch.finfo(mag.dtype).eps)


def orthogonal(measured_magnitude, mask=None, n_trials=5, generator=None):
    """Deprecated name for random_phase_multistart; no orthogonality constraint."""
    warnings.warn('orthogonal is a random-phase search; use random_phase_multistart',
                  DeprecationWarning, stacklevel=2)
    return random_phase_multistart(measured_magnitude, mask, n_trials, generator)


def power_method(y, A=None, At=None, n_iter: int = 50):
    """
    Experimental power iteration on A* diag(y) A, normalized per image.

    y is a nonnegative measurement weight (usually squared magnitudes). With
    a single unitary Fourier measurement, this operator is diagonal in Fourier
    space and does not provide a general phase-recovery guarantee.

    Args:
        y: nonnegative measurement weights (tensor)
        A: forward operator (callable), defaults to FFT2
        At: adjoint operator (callable), defaults to iFFT2
        n_iter: number of power iterations

    Returns:
        v: leading eigenvector (complex tensor)
    """
    if (A is None) != (At is None):
        raise ValueError('Supply both the forward operator and its adjoint')
    if A is None:
        A, At = _default_ops(y.shape)

    # random complex init
    v = torch.randn(y.shape, dtype=torch.complex128 if y.dtype == torch.float64 else torch.complex64, device=y.device)

    for _ in range(n_iter):
        Av = A(v)
        v = At(y * Av)
        v = transforms.normalize_field(v)

    return v


def wirtinger_flow(y, A=None, At=None, n_iter: int = 200, step_size: float = 0.1):
    """
    Experimental fixed-step intensity-gradient phase retrieval.

    Convergence depends on the supplied measurement operator and step size;
    no guarantee is implied for a single Fourier-magnitude measurement.

    Args:
        y: measurement magnitudes
        A: forward operator, defaults to FFT2
        At: adjoint operator, defaults to iFFT2
        n_iter: iterations
        step_size: gradient descent step size

    Returns:
        x: recovered signal
    """
    if (A is None) != (At is None):
        raise ValueError('Supply both the forward operator and its adjoint')
    if A is None:
        A, At = _default_ops(y.shape)

    # spectral initialization
    x = power_method(y**2, A, At, n_iter=50)

    for _ in range(n_iter):
        Ax = A(x)
        residual = (torch.abs(Ax)**2 - y**2) * Ax
        grad = At(residual)
        x = x - step_size * grad

    return x


def gao_xu(y, A=None, At=None, n_iter: int = 200, step_size: float = 0.1, tau: float = 3.0):
    """
    Experimental truncated intensity-gradient heuristic, not amplitude flow.

    Args:
        y: measurement magnitudes
        A: forward operator, defaults to FFT2
        At: adjoint operator, defaults to iFFT2
        n_iter: iterations
        step_size: learning rate
        tau: truncation threshold factor

    Returns:
        x: recovered signal
    """
    if (A is None) != (At is None):
        raise ValueError('Supply both the forward operator and its adjoint')
    if A is None:
        A, At = _default_ops(y.shape)

    # spectral initialization
    x = power_method(y**2, A, At, n_iter=50)

    for _ in range(n_iter):
        Ax = A(x)
        residual = (torch.abs(Ax)**2 - y**2) * Ax

        # truncation
        mask = torch.abs(residual) < tau * y
        residual = residual * mask

        grad = At(residual)
        x = x - step_size * grad

    return x


def gradient_descent(
        magnitude_near_field, magnitude_far_field, support=None,
        init_phase=None, optimizer_class=torch.optim.Adam, optimizer_kwargs=None,
        max_iter=50, cyclic_loss=False, return_loss=False, *,
        loss_type='amplitude_mse', initialization='flat', generator=None,
        log_every=0, metric_every=1, scheduler_factory=None, scheduler_every=10):
    """Optimize fixed near-field amplitude and phase on NCHW tensors.

    Loss is mean per-image normalized amplitude MSE by default. Alternatives:
    intensity_mse, pearson_amplitude, pearson_intensity. History includes the
    initial field and the final returned field; it is transferred to CPU once.
    A supplied scheduler_factory(optimizer) must return ReduceLROnPlateau or a
    scheduler with step(); no host-synchronizing scheduler is enabled by default.
    """
    if max_iter < 0 or metric_every < 1 or log_every < 0 or scheduler_every < 1:
        raise ValueError('Invalid iteration or logging interval')
    near, far = magnitude_near_field.abs(), magnitude_far_field.abs()
    if near.ndim != 4 or near.shape != far.shape:
        raise ValueError('Expected matching NCHW magnitude tensors')
    if init_phase is None:
        if initialization == 'flat':
            init_phase = torch.zeros_like(near)
        elif initialization == 'random':
            init_phase = _random_phases_from_tensor(near, generator)
        else:
            raise ValueError("initialization must be 'flat' or 'random'")
    model = PhaseOnlyOptimizer(near, init_phase, support, loss_type=loss_type)
    optimizer = optimizer_class(model.parameters(), **({'lr': 0.1} if optimizer_kwargs is None else dict(optimizer_kwargs)))
    scheduler = None if scheduler_factory is None else scheduler_factory(optimizer)
    history = []
    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(far, near if cyclic_loss else None)
        # Reuse the training forward pass for the sampled current iterate.
        if return_loss and i % metric_every == 0:
            history.append(loss.detach())
        loss.backward()
        optimizer.step()
        step = i + 1
        if scheduler is not None and step % scheduler_every == 0:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.detach().item())
            else:
                scheduler.step()
        if log_every and step % log_every == 0:
            print(f'Iteration {step}, Loss: {loss.detach().item():.6g}')
    with torch.no_grad():
        output = model().detach()
        if return_loss:
            history.append(model.loss(far, near if cyclic_loss else None).detach())
    if return_loss:
        return output, torch.stack(history).cpu().tolist()
    return output


class PhaseOnlyOptimizer(nn.Module):
    def __init__(self, init_magnitude, init_phase, support=None,
                 loss_type='amplitude_mse'):
        super().__init__()
        if init_magnitude.ndim != 4 or init_phase.shape != init_magnitude.shape:
            raise ValueError('Expected matching NCHW magnitude and phase')
        if loss_type not in ('amplitude_mse', 'intensity_mse', 'pearson_amplitude', 'pearson_intensity'):
            raise ValueError('Unknown loss_type')
        self.register_buffer('magnitude', init_magnitude.abs().detach().clone())
        self.register_buffer('support', torch.ones_like(init_magnitude, dtype=torch.bool)
                             if support is None else support.to(device=init_magnitude.device, dtype=torch.bool))
        self.phase = nn.Parameter(init_phase.to(device=init_magnitude.device,
                                               dtype=init_magnitude.real.dtype).detach().clone())
        self.loss_type = loss_type

    def forward(self):
        return transforms.apply_support(torch.polar(self.magnitude, self.phase), self.support)

    def _loss_per_image(self, x, y):
        x, y = x.abs(), y.abs()
        if self.loss_type in ('intensity_mse', 'pearson_intensity'):
            x, y = x.square(), y.square()
        if self.loss_type.startswith('pearson'):
            return metrics.pearson(x, y, inversed=True)
        return metrics.normalized_mse(x, y)

    def loss(self, target_fourier_magnitude, target_magnitude=None):
        x_fft = transforms.fourier_transform(self())
        loss = self._loss_per_image(x_fft, target_fourier_magnitude)
        if target_magnitude is not None:
            constrained = target_fourier_magnitude.abs() * torch.exp(1j * x_fft.angle())
            back = transforms.inverse_fourier_transform(constrained)
            target = transforms.apply_support(target_magnitude.abs(), self.support)
            loss = 0.5 * (loss + self._loss_per_image(back, target))
        return loss.mean()


class PearsonImageLoss(nn.Module):
    """Compatibility wrapper around the shared population Pearson metric."""
    def __init__(self, average_over_channels=True, average_over_batches=True,
                 invert=True, power=1):
        super().__init__()
        self.average_over_channels = average_over_channels
        self.average_over_batches = average_over_batches
        self.invert, self.power = invert, power

    def forward(self, x, y):
        return batch_pearson_images(x.abs().pow(self.power), y.abs().pow(self.power),
                                   self.average_over_channels, self.average_over_batches,
                                   self.invert)


def batch_pearson_images(x, y, average_over_channels=True,
                         average_over_batches=True, invert=False):
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError('Expected matching NCHW images')
    corr = metrics.pearson(x.abs(), y.abs(), inversed=invert)[..., None, None]
    if average_over_channels:
        corr = corr.mean(dim=1, keepdim=True)
    if average_over_batches:
        corr = corr.mean()
    return corr

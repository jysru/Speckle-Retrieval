"""Automatic-differentiation reconstruction and projection-to-AD refinement.

All objectives reduce spatial dimensions only. First-order optimization sums
independent per-image objectives, avoiding batch-size-dependent SGD step sizes.
Stock L-BFGS uses one shared line search over the whole batch.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from . import metrics, transforms


LOSS_TYPES = ('amplitude_mse', 'intensity_mse', 'huber_amplitude',
              'huber_intensity', 'pearson_amplitude', 'pearson_intensity', 'poisson')


def _pair(value, name):
    if value is None or not hasattr(value, '__len__') or len(value) != 2:
        raise ValueError(f'{name} must contain near and far plane values')
    return tuple(value)


class MeasurementLoss(nn.Module):
    """Fixed detector observations and a differentiable per-N,C data fidelity.

    input_type: amplitude (field magnitude), intensity (sensor intensity), or
    counts (raw nonnegative photon counts). Expected intensity/counts are
    gain*abs(field)**2 + background. Masks and nonnegative confidence weights
    broadcast to NCHW. Masked NaNs are ignored; observed NaNs are rejected.

    poisson is the generalized KL divergence (Poisson NLL minus its data-only
    minimum), averaged over observed weights. It requires counts, not normalized
    amplitude data. Other MSE losses are normalized by weighted target energy;
    zero targets use unscaled residuals. Huber delta is in normalized units.
    """
    def __init__(self, target, *, input_type='amplitude', loss_type='amplitude_mse',
                 mask=None, weights=None, gain=1., background=0., huber_delta=1.):
        super().__init__()
        if input_type not in ('amplitude', 'intensity', 'counts') or loss_type not in LOSS_TYPES:
            raise ValueError('Unknown input_type or loss_type')
        if loss_type == 'poisson' and input_type != 'counts':
            raise ValueError('Poisson loss requires raw counts with input_type="counts"')
        if target.ndim != 4 or torch.is_complex(target):
            raise ValueError('Measurements must be real NCHW tensors')
        if not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError('huber_delta must be positive and finite')
        dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
        target = target.detach().to(dtype=dtype)

        def tensor(value):
            return torch.broadcast_to(torch.as_tensor(value, device=target.device, dtype=dtype), target.shape).clone().detach()

        valid = torch.ones_like(target, dtype=torch.bool) if mask is None else torch.broadcast_to(
            torch.as_tensor(mask, device=target.device, dtype=torch.bool), target.shape)
        weight = tensor(1. if weights is None else weights)
        weight = torch.where(valid, weight, 0)
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('Observed weights must be finite and nonnegative')
        valid = valid & (weight > 0)
        target = torch.where(valid, target, 0)
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError('Observed measurements must be finite and nonnegative')
        def calibration(value):
            value = torch.as_tensor(value, device=target.device, dtype=dtype)
            torch.broadcast_to(value, target.shape)  # Validate without materializing scalars.
            return value.clone().detach()

        gain, background = calibration(gain), calibration(background)
        if not torch.isfinite(gain).all() or (gain <= 0).any():
            raise ValueError('Detector gain must be positive and finite')
        if not torch.isfinite(background).all() or (background < 0).any():
            raise ValueError('Detector background must be nonnegative and finite')
        self.register_buffer('target', target.square() if input_type == 'amplitude' else target)
        self.register_buffer('weight', torch.where(valid, weight, 0))
        self.register_buffer('observed', valid.clone())
        total = self.weight.sum((-2, -1))
        self.register_buffer('normalizer', torch.where(total > 0, total, 1))
        self.register_buffer('gain', gain)
        self.register_buffer('background', background)
        self.input_type, self.loss_type, self.huber_delta = input_type, loss_type, huber_delta
        target_values = self.target if loss_type in ('poisson', 'intensity_mse', 'huber_intensity', 'pearson_intensity') else self.target.sqrt()
        self.register_buffer('target_values', target_values)
        energy = self._mean(target_values.square())
        self.register_buffer('target_scale', torch.where(energy > 0, energy, 1))
        self.register_buffer('poisson_data_log', torch.xlogy(self.target, self.target.clamp_min(torch.finfo(dtype).tiny))
                             if loss_type == 'poisson' else None)

    @property
    def valid(self):
        return self.observed

    def initial_amplitude(self):
        """Background-corrected amplitude for initialization, not the likelihood."""
        return ((self.target - self.background) / self.gain).clamp_min(0).sqrt()

    def _mean(self, value):
        # where also suppresses masked NaNs in a supplied prediction.
        value = torch.where(self.valid, value, 0)
        return (value * self.weight).sum((-2, -1)) / self.normalizer

    def forward(self, field):
        if field.shape != self.target.shape:
            raise ValueError('Forward-model output shape must match its measurements')
        rate = self.gain * field.abs().square() + self.background
        target = self.target
        if self.loss_type == 'poisson':
            # Stable at zero counts and at zero predicted rate; no log(0).
            safe = rate.clamp_min(torch.finfo(rate.dtype).eps)
            return self._mean(rate - target + self.poisson_data_log - target * safe.log())
        if self.loss_type.endswith('intensity') or self.loss_type == 'intensity_mse':
            predicted = rate
        else:
            # Preserve NaNs from a broken forward model instead of treating
            # them as dark pixels; only the exact-zero derivative needs fixing.
            predicted = torch.where(rate == 0, 0, rate.clamp_min(torch.finfo(rate.dtype).tiny).sqrt())
        target = self.target_values
        if self.loss_type.startswith('pearson'):
            dx = predicted - self._mean(predicted)[..., None, None]
            dy = target - self._mean(target)[..., None, None]
            vx, vy = self._mean(dx.square()), self._mean(dy.square())
            tiny = torch.finfo(rate.dtype).tiny
            denominator = vx.clamp_min(tiny).sqrt() * vy.clamp_min(tiny).sqrt()
            corr = torch.where((vx > 0) & (vy > 0), self._mean(dx*dy) / denominator.clamp_min(tiny), 0)
            loss = torch.where(torch.isfinite(vx) & torch.isfinite(vy),
                                1-corr.clamp(-1, 1), float('nan'))
            return torch.where(self.weight.sum((-2, -1)) > 0, loss, 0)
        scale = self.target_scale
        if self.loss_type.startswith('huber'):
            residual = (predicted-target) / scale.sqrt()[..., None, None]
            return self._mean(F.huber_loss(residual, torch.zeros_like(residual),
                                            delta=self.huber_delta, reduction='none'))
        return self._mean((predicted-target).square()) / scale


class PhaseOnlyOptimizer(nn.Module):
    """Supported field with measured amplitude and learnable real-valued phase.

    forward_model/inverse_model are fixed callables. Only field parameters are
    optimized; captured calibration parameters are not managed by this class.
    """
    def __init__(self, init_magnitude, init_phase, support=None,
                 loss_type='amplitude_mse', forward_model=None, inverse_model=None):
        super().__init__()
        if init_magnitude.ndim != 4 or init_phase.shape != init_magnitude.shape:
            raise ValueError('Expected matching NCHW magnitude and phase')
        if loss_type not in LOSS_TYPES:
            raise ValueError('Unknown loss_type')
        self.register_buffer('magnitude', init_magnitude.abs().detach().clone())
        self.register_buffer('support', torch.ones_like(init_magnitude, dtype=torch.bool)
                             if support is None else torch.as_tensor(support, device=init_magnitude.device, dtype=torch.bool))
        self.phase = nn.Parameter(init_phase.to(device=init_magnitude.device,
                                               dtype=init_magnitude.real.dtype).detach().clone())
        self.loss_type = loss_type
        # Do not accidentally register/optimize a user-supplied operator's weights.
        object.__setattr__(self, 'forward_operator', transforms.fourier_transform if forward_model is None else forward_model)
        object.__setattr__(self, 'inverse_operator', transforms.inverse_fourier_transform if forward_model is None and inverse_model is None else inverse_model)

    def forward(self):
        return transforms.apply_support(torch.polar(self.magnitude, self.phase), self.support)

    def _loss_per_image(self, x, y):
        # Historical model.loss interface: amplitude inputs, no detector model.
        x, y = x.abs(), y.abs()
        if self.loss_type in ('intensity_mse', 'pearson_intensity'):
            x, y = x.square(), y.square()
        if self.loss_type.startswith('pearson'):
            return metrics.pearson(x, y, inversed=True)
        if self.loss_type not in ('amplitude_mse', 'intensity_mse'):
            raise ValueError('Use MeasurementLoss/reconstruct for robust or count losses')
        return metrics.normalized_mse(x, y)

    def loss(self, target_fourier_magnitude, target_magnitude=None):
        spectrum = self.forward_operator(self())
        loss = self._loss_per_image(spectrum, target_fourier_magnitude)
        if target_magnitude is not None:
            if self.inverse_operator is None:
                raise ValueError('Cyclic loss needs an explicit inverse_model for custom propagation')
            constrained = target_fourier_magnitude.abs() * torch.exp(1j * spectrum.angle())
            back = self.inverse_operator(constrained)
            target = transforms.apply_support(target_magnitude.abs(), self.support)
            loss = .5 * (loss + self._loss_per_image(back, target))
        return loss.mean()


class ComplexFieldOptimizer(nn.Module):
    """Supported Cartesian complex field; both real and imaginary parts can move."""
    def __init__(self, init_field, support=None):
        super().__init__()
        if init_field.ndim != 4 or not torch.is_complex(init_field):
            raise ValueError('ComplexFieldOptimizer expects a complex NCHW field')
        self.real = nn.Parameter(init_field.real.detach().clone())
        self.imag = nn.Parameter(init_field.imag.detach().clone())
        self.register_buffer('support', torch.ones_like(init_field.real, dtype=torch.bool)
                             if support is None else torch.as_tensor(support, device=init_field.device, dtype=torch.bool))

    def forward(self):
        return transforms.apply_support(torch.complex(self.real, self.imag), self.support)


def _terms(measurements, input_type, loss_type, masks, weights, gains, backgrounds, huber_delta):
    observations = _pair(measurements, 'measurements')
    if any(m.ndim != 4 or torch.is_complex(m) for m in observations):
        raise ValueError('Measurements must be real NCHW tensors')
    if observations[0].device != observations[1].device or observations[0].shape[:2] != observations[1].shape[:2]:
        raise ValueError('Both measurements must share device and N,C dimensions')
    dtype = torch.float64 if any(m.dtype == torch.float64 for m in observations) else torch.float32
    options = zip(observations, _pair(masks, 'masks'), _pair(weights, 'weights'),
                  _pair(gains, 'gains'), _pair(backgrounds, 'backgrounds'))
    return tuple(MeasurementLoss(m.to(dtype=dtype), input_type=input_type, loss_type=loss_type,
                                 mask=mask, weights=weight, gain=gain, background=background,
                                 huber_delta=huber_delta)
                 for m, mask, weight, gain, background in options)


def _initial_amplitude(term):
    amplitude = term.initial_amplitude()
    count = term.valid.sum((-2, -1), keepdim=True)
    mean = torch.where(term.valid, amplitude, 0).sum((-2, -1), keepdim=True) / count.clamp_min(1)
    # Missing near pixels need an initial guess rather than NaNs; amplitude
    # optimization is required if those guessed values should be learned.
    return torch.where(term.valid, amplitude, torch.where(count > 0, mean, 1))


def reconstruct(measurements, support=None, *, input_type='amplitude',
                loss_type='amplitude_mse', optimize_amplitude=False,
                init=None, init_phase=None, initialization='flat', generator=None,
                masks=(None, None), weights=(None, None), gains=(1., 1.),
                backgrounds=(0., 0.), plane_weights=None, huber_delta=1.,
                forward_model=None, inverse_model=None, cyclic_loss=False,
                optimizer_class=torch.optim.Adam, optimizer_kwargs=None,
                max_iter=100, metric_every=10, return_best=True,
                rel_tol=None, patience=5, check_every=10,
                scheduler_factory=None, scheduler_every=10, log_every=0):
    """Fit a phase-only or complex field to two detector measurements.

    Returns (supported_field, diagnostics). Histories include initialization,
    sampled updates, and the final returned field; best selection is per N,C.
    Default objective is far-plane only for phase-only fitting, and equal near/
    far fidelity when amplitude is learned. Custom operators must be deterministic
    and act independently on each image/channel. Far spatial size may differ.

    L-BFGS closures can evaluate the objective several times per outer update;
    function_evaluations includes closures and diagnostic forward evaluations.
    Early stopping, when enabled, checks all images at check_every intervals and
    stops only after every image stalls for patience checks. Checks synchronize
    the device; the default fixed-iteration loop does not check host scalars.
    """
    if max_iter < 0 or metric_every < 1 or log_every < 0 or scheduler_every < 1 or patience < 1 or check_every < 1:
        raise ValueError('Invalid iteration count or interval')
    if rel_tol is not None and (not math.isfinite(rel_tol) or rel_tol < 0):
        raise ValueError('rel_tol must be finite and nonnegative')
    for operator in (forward_model, inverse_model):
        if isinstance(operator, nn.Module) and any(p.requires_grad for p in operator.parameters()):
            raise ValueError('Freeze operator parameters; reconstruct optimizes field parameters only')
    near, far = _terms(measurements, input_type, loss_type, masks, weights, gains, backgrounds, huber_delta)
    plane_weights = (1., 1.) if plane_weights is None and optimize_amplitude else ((0., 1.) if plane_weights is None else _pair(plane_weights, 'plane_weights'))
    if any(not math.isfinite(w) or w < 0 for w in plane_weights) or sum(plane_weights) == 0:
        raise ValueError('Plane weights must be finite, nonnegative, and not both zero')
    observed = (near.weight.sum((-2, -1)) > 0) * plane_weights[0] + (far.weight.sum((-2, -1)) > 0) * plane_weights[1]
    if (observed == 0).any():
        raise ValueError('Every image/channel needs at least one observed, weighted plane')
    amplitude = _initial_amplitude(near)
    if init is not None:
        if init.shape != amplitude.shape or init.device != amplitude.device:
            raise ValueError('Initialization must match near-field shape and device')
        if init_phase is not None:
            raise ValueError('Supply init or init_phase, not both')
        phase = init.angle().to(amplitude.dtype)
    elif init_phase is not None:
        phase = init_phase.to(device=amplitude.device, dtype=amplitude.dtype)
    elif initialization == 'flat':
        phase = torch.zeros_like(amplitude)
    elif initialization == 'random':
        phase = 2*torch.pi*torch.rand(amplitude.shape, device=amplitude.device,
                                     dtype=amplitude.dtype, generator=generator) - torch.pi
    else:
        raise ValueError('initialization must be flat or random')
    if phase.shape != amplitude.shape:
        raise ValueError('Phase initialization must match near-field shape')
    forward = transforms.fourier_transform if forward_model is None else forward_model
    inverse = transforms.inverse_fourier_transform if forward_model is None and inverse_model is None else inverse_model
    if cyclic_loss and (inverse is None or input_type != 'amplitude' or optimize_amplitude):
        raise ValueError('Cyclic loss requires phase-only amplitude data and an inverse_model')
    if cyclic_loss and (any(m is not None for m in masks) or any(w is not None for w in weights)
                        or any(torch.as_tensor(g).ne(1).any() for g in gains)
                        or any(torch.as_tensor(b).ne(0).any() for b in backgrounds)):
        raise ValueError('Cyclic loss currently requires complete unweighted calibrated amplitudes')
    if optimize_amplitude:
        initial = torch.polar(amplitude, phase) if init is None else init.to(dtype=torch.complex128 if amplitude.dtype == torch.float64 else torch.complex64)
        if init is None:
            # A completely zero Cartesian field is stationary for intensity
            # objectives even when the far detector sees light. Seed those
            # images from measured energy instead of silently getting stuck.
            object_mask = torch.ones_like(amplitude, dtype=torch.bool) if support is None else torch.broadcast_to(
                torch.as_tensor(support, device=amplitude.device, dtype=torch.bool), amplitude.shape)
            empty = transforms.apply_support(initial, object_mask).abs().square().sum((-2, -1), keepdim=True) == 0
            energy = far.initial_amplitude().square().sum((-2, -1), keepdim=True)
            seed = (energy / object_mask.sum((-2, -1), keepdim=True).clamp_min(1)).sqrt()
            initial = torch.where(empty, seed * torch.exp(1j*phase), initial)
        model = ComplexFieldOptimizer(initial, support)
    else:
        model = PhaseOnlyOptimizer(amplitude, phase, support)
    evaluations = 0

    def evaluate():
        nonlocal evaluations
        evaluations += 1
        field = model()
        propagated = forward(field) if plane_weights[1] or cyclic_loss else None
        value = 0
        if plane_weights[0]:
            value = value + plane_weights[0] * near(field)
        if plane_weights[1]:
            value = value + plane_weights[1] * far(propagated)
        value = value / sum(plane_weights)
        if cyclic_loss:
            constrained = far.target.sqrt() * torch.exp(1j * propagated.angle())
            back = inverse(constrained)
            # Historical cyclic penalty compares the entire back-propagated
            # field against the supported object amplitude.
            target = transforms.apply_support(amplitude, model.support)
            value = .5 * (value + metrics.normalized_mse(back, target))
        return field, value

    options = dict(optimizer_kwargs or {})
    is_lbfgs = issubclass(optimizer_class, torch.optim.LBFGS)
    if is_lbfgs:
        options = {'lr': 1., 'max_iter': 10, 'history_size': 10,
                   'line_search_fn': 'strong_wolfe', **options}
    else:
        if optimizer_kwargs is None:
            options['lr'] = .1
    optimizer = optimizer_class(model.parameters(), **options)
    scheduler = None if scheduler_factory is None else scheduler_factory(optimizer)
    with torch.set_grad_enabled(not is_lbfgs):
        current, value = evaluate()
    initial_loss = value.detach().clone()
    if not torch.isfinite(initial_loss).all():
        raise ValueError('Initial reconstruction objective is not finite; check the field and forward model')
    best, best_loss = current.detach().clone(), initial_loss.clone()
    best_step = torch.zeros_like(best_loss, dtype=torch.long)
    previous_check = best_loss.clone()
    stalled = torch.zeros_like(best_step)
    history, steps = [initial_loss], [0]
    completed = 0

    for i in range(max_iter):
        if is_lbfgs:
            # Never reuse an old graph after L-BFGS changes parameters.
            def closure():
                optimizer.zero_grad(set_to_none=True)
                _, losses = evaluate()
                scalar = losses.sum()
                scalar.backward()
                return scalar
            optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            value.sum().backward()
            optimizer.step()
        completed = i + 1
        with torch.set_grad_enabled(not is_lbfgs):
            current, value = evaluate()
        detached = value.detach()
        better = detached < best_loss
        best = torch.where(better[..., None, None], current.detach(), best)
        best_step = torch.where(better, completed, best_step)
        best_loss = torch.minimum(detached, best_loss)
        if completed % metric_every == 0 or completed == max_iter:
            history.append(detached.clone())
            steps.append(completed)
        if scheduler is not None and completed % scheduler_every == 0:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(detached.mean().item())
            else:
                scheduler.step()
        if log_every and completed % log_every == 0:
            print(f'Iteration {completed}, mean loss: {detached.mean().item():.6g}')
        if rel_tol is not None and completed % check_every == 0:
            threshold = rel_tol * previous_check.abs().clamp_min(torch.finfo(value.dtype).eps)
            improved = previous_check - best_loss > threshold
            stalled = torch.where(improved, 0, stalled + 1)
            previous_check = best_loss.clone()
            if (stalled >= patience).all().item():
                break
    output = best if return_best else current.detach()
    final_loss = best_loss if return_best else value.detach()
    if steps[-1] != completed:
        steps.append(completed)
        history.append(final_loss.clone())
    else:
        history[-1] = final_loss.clone()
    return output, {'loss': torch.stack(history), 'iterations': torch.tensor(steps, device=output.device),
                    'initial_loss': initial_loss, 'final_loss': final_loss,
                    'best_loss': best_loss, 'best_iteration': best_step,
                    'iterations_completed': completed, 'function_evaluations': evaluations,
                    'stalled': stalled >= patience, 'optimizer': optimizer_class.__name__}


def gradient_descent(magnitude_near_field, magnitude_far_field, support=None,
                     init_phase=None, optimizer_class=torch.optim.Adam, optimizer_kwargs=None,
                     max_iter=50, cyclic_loss=False, return_loss=False, *,
                     loss_type='amplitude_mse', initialization='flat', generator=None,
                     log_every=0, metric_every=1, scheduler_factory=None, scheduler_every=10,
                     **kwargs):
    """Compatibility amplitude API; returns a field or (field, scalar-loss list)."""
    output, info = reconstruct((magnitude_near_field.abs(), magnitude_far_field.abs()), support,
                               init_phase=init_phase, optimizer_class=optimizer_class,
                               optimizer_kwargs=optimizer_kwargs, max_iter=max_iter,
                               cyclic_loss=cyclic_loss, loss_type=loss_type,
                               initialization=initialization, generator=generator,
                               log_every=log_every, metric_every=metric_every,
                               scheduler_factory=scheduler_factory, scheduler_every=scheduler_every,
                               return_best=kwargs.pop('return_best', False), **kwargs)
    return (output, info['loss'].mean(dim=(1, 2)).cpu().tolist()) if return_loss else output


def hybrid_reconstruct(measurements, support=None, *, n_restarts=4,
                       projection_iter=100, refinement_iter=100, generator=None,
                       projection_beta=.9, optimizer_class=torch.optim.Adam,
                       optimizer_kwargs=None, metric_every=10, **kwargs):
    """Batched RAAR starts -> per-image objective selection -> AD refinement.

    All projection candidates are ranked using the SAME masked/weighted detector
    objective as refinement, not a reference field or unmasked Fourier MSE.
    Missing detector pixels remain unconstrained during RAAR; uncertain near
    amplitudes can subsequently move with optimize_amplitude=True. Projection
    uses calibrated amplitudes even when observations are intensities/counts.

    The initializer currently requires a Fourier model and equal plane sizes.
    For custom propagation, supply your initial field directly to reconstruct().
    Return_best is enabled by default, so refinement cannot worsen the selected
    candidate's objective. L-BFGS refines selected candidates jointly, with its
    shared line search; it is not an independently vectorized L-BFGS solver.
    """
    from . import retrieval

    if n_restarts < 1 or projection_iter < 0 or refinement_iter < 0:
        raise ValueError('Invalid restart count or iteration count')
    if kwargs.get('forward_model') is not None or kwargs.get('inverse_model') is not None:
        raise ValueError('Hybrid RAAR initialization requires FFT propagation; use reconstruct(init=...) for custom models')
    if any(key in kwargs for key in ('init', 'init_phase', 'initialization', 'max_iter')):
        raise ValueError('Hybrid initialization and iteration counts are managed by the pipeline')
    if kwargs.get('cyclic_loss', False):
        raise ValueError('Hybrid selection currently uses direct measurement losses, not cyclic_loss')
    input_type, loss_type = kwargs.get('input_type', 'amplitude'), kwargs.get('loss_type', 'amplitude_mse')
    options = dict(masks=kwargs.get('masks', (None, None)), weights=kwargs.get('weights', (None, None)),
                   gains=kwargs.get('gains', (1., 1.)), backgrounds=kwargs.get('backgrounds', (0., 0.)))
    huber_delta = kwargs.get('huber_delta', 1.)
    near, far = _terms(measurements, input_type, loss_type, **options, huber_delta=huber_delta)
    if near.target.shape != far.target.shape:
        raise ValueError('Hybrid FFT initialization requires equal near/far spatial sizes')
    n, c, h, w = near.target.shape

    def expand(tensor):
        tensor = torch.broadcast_to(tensor, near.target.shape)
        return tensor[:, None].expand(n, n_restarts, c, h, w).reshape(n*n_restarts, c, h, w)

    # Rebuild fixed detector terms in flattened N*R order; targets are already
    # intensities, so take square roots only for the amplitude-domain interface.
    def expand_term(term):
        observation = term.target.sqrt() if input_type == 'amplitude' else term.target
        return MeasurementLoss(expand(observation), input_type=input_type, loss_type=loss_type,
                               weights=expand(term.weight), gain=expand(term.gain),
                               background=expand(term.background), huber_delta=huber_delta)

    expanded_near, expanded_far = expand_term(near), expand_term(far)
    near_amp, far_amp = _initial_amplitude(expanded_near), expanded_far.initial_amplitude()
    mask = torch.ones_like(near_amp, dtype=torch.bool) if support is None else expand(torch.as_tensor(
        support, device=near_amp.device, dtype=torch.bool))
    plane_weights = kwargs.get('plane_weights')
    optimize_amplitude = kwargs.get('optimize_amplitude', False)
    if plane_weights is None:
        plane_weights = (1., 1.) if optimize_amplitude else (0., 1.)
    plane_weights = _pair(plane_weights, 'plane_weights')
    if any(not math.isfinite(w) or w < 0 for w in plane_weights) or sum(plane_weights) == 0:
        raise ValueError('Plane weights must be finite, nonnegative and not both zero')

    def object_candidate(field):
        if not optimize_amplitude:
            field = near_amp * torch.exp(1j * field.angle())
        return transforms.apply_support(field, mask)

    def objective(field):
        field = object_candidate(field)
        value = 0
        if plane_weights[0]:
            value = value + plane_weights[0]*expanded_near(field)
        if plane_weights[1]:
            value = value + plane_weights[1]*expanded_far(transforms.fourier_transform(field))
        return value / sum(plane_weights)

    def project_far(field, magnitude):
        spectrum = transforms.fourier_transform(field)
        constrained = magnitude * torch.exp(1j * spectrum.angle())
        return transforms.inverse_fourier_transform(torch.where(expanded_far.valid, constrained, spectrum))

    def reflect_near(field, gamma, magnitude, support):
        constrained = magnitude * torch.exp(1j * field.angle())
        projection = transforms.apply_support(torch.where(expanded_near.valid, constrained, field), support)
        return (1+gamma)*projection - gamma*field

    phase = 2*torch.pi*torch.rand(near_amp.shape, device=near_amp.device,
                                 dtype=near_amp.dtype, generator=generator) - torch.pi
    with torch.no_grad():
        raw, projection_info = retrieval.run(
            (near_amp, far_amp), mask, init=torch.polar(near_amp, phase),
            algorithm=retrieval.RAAR,
            algorithm_kwargs={'beta': projection_beta, 'projector_M': project_far,
                              'reflector_S': reflect_near},
            max_iter=projection_iter, metric_every=metric_every,
            return_best=True, selection_loss=objective)
        candidates = object_candidate(raw).reshape(n, n_restarts, c, h, w)
        scores = projection_info['best_selection_loss'].reshape(n, n_restarts, c)
        selected_loss, selected = scores.min(dim=1)
        index = selected[:, None, :, None, None].expand(n, 1, c, h, w)
        initial = candidates.gather(1, index).squeeze(1)
    field, refinement_info = reconstruct(
        measurements, support, init=initial, optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs, max_iter=refinement_iter,
        metric_every=metric_every, **kwargs)
    return field, {'selected_restart': selected, 'projection_loss': selected_loss,
                    'projection': projection_info, 'refinement': refinement_info}

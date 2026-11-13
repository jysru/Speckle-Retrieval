"""Batched projection algorithms and measurement-based reconstruction tracking."""
from functools import lru_cache, partial
from . import supports

import torch
from . import metrics
from . import transforms
from . import initializers


# Review paper: https://doi.org/10.1063/1.2403783
# Google collab: https://colab.research.google.com/drive/1anePjgg1fKbYrCCmDeRKblobryq-O4Rv


def P_S(
        x: torch.Tensor,
        magnitude: torch.Tensor,
        support: torch.Tensor,
        apply_support: bool = True,
    ) -> torch.Tensor:
    """
    Support projection for complex tensor [N, C, H, W].

    Args:
        x: complex input tensor [N, C, H, W]
        magnitude: real tensor [N, C, H, W] with desired magnitudes
        support: boolean mask tensor [N, C, H, W]
        apply_support: whether to zero outside support

    Returns:
        projected complex tensor [N, C, H, W]
    """
    if not torch.is_complex(x):
        raise ValueError("Input x must be complex tensor")
    if magnitude.shape != x.shape:
        raise ValueError("Magnitude shape must match input shape")

    # Compute new field with original phase, new magnitude
    phase = torch.angle(x)
    x_new = magnitude * torch.exp(1j * phase)

    if apply_support:
        x_new = transforms.apply_support(x_new, support)

    return x_new


def P_M(
        x: torch.Tensor,
        magnitude: torch.Tensor,
        direct_transform: callable = transforms.fourier_transform,
        inverse_transform: callable = transforms.inverse_fourier_transform,
    ) -> torch.Tensor:
    """
    Fourier magnitude projection for complex-valued tensors [N, C, H, W].

    Args:
        x: Complex input tensor.
        magnitude: Desired magnitude tensor of same shape as x.
        direct_transform: Callable performing forward transform (e.g., FFT).
        inverse_transform: Callable performing inverse transform (e.g., IFFT).

    Returns:
        Tensor after enforcing magnitude constraint in Fourier domain.
    """
    if not torch.is_complex(x):
        raise ValueError("Input x must be a complex tensor")
    if magnitude.shape != x.shape:
        raise ValueError("Magnitude shape must match input shape")

    # Forward transform (Fourier domain)
    X = direct_transform(x)

    # Enforce magnitude, keep phase
    phase = torch.angle(X)
    X_new = magnitude * torch.exp(1j * phase)

    # Inverse transform back to spatial domain
    x_proj = inverse_transform(X_new)
    return x_proj


def R_S(
        x: torch.Tensor,
        gamma_S: float,
        magnitude: torch.Tensor,
        support: torch.Tensor,
        projector_S: callable = P_S,
    ) -> torch.Tensor:
    """
    Support reflector.

    Args:
        x: Complex input tensor [N,C,H,W].
        gamma_S: Relaxation parameter (float).
        magnitude: Desired magnitude tensor [N,C,H,W].
        support: Boolean support mask tensor [N,C,H,W].
        projector_S: Callable projector function, e.g. P_S.

    Returns:
        Reflected tensor [N,C,H,W].
    """
    proj = projector_S(x, magnitude, support)
    return (1 + gamma_S) * proj - gamma_S * x


def R_M(
        x: torch.Tensor,
        gamma_M: float,
        magnitude: torch.Tensor,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Magnitude reflector.

    Args:
        x: Complex input tensor [N,C,H,W].
        gamma_M: Relaxation parameter (float).
        magnitude: Desired magnitude tensor [N,C,H,W].
        projector_M: Callable projector function, e.g. P_M.

    Returns:
        Reflected tensor [N,C,H,W].
    """
    # Apply the magnitude projector
    proj = projector_M(x, magnitude)
    # Reflector formula
    return (1 + gamma_M) * proj - gamma_M * x


def ER(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        projector_S: callable = P_S,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Error Reduction algorithm iteration.

    Args:
        x: Complex input tensor [N,C,H,W].
        magnitude_S: Desired support magnitude [N,C,H,W].
        magnitude_M: Desired Fourier magnitude [N,C,H,W].
        support: Boolean support mask [N,C,H,W].
        projector_S: Support projector callable.
        projector_M: Magnitude projector callable.

    Returns:
        Updated tensor after ER iteration.
    """
    x_proj_M = projector_M(x, magnitude_M)
    x_proj_S = projector_S(x_proj_M, magnitude_S, support)
    return x_proj_S


def SF(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        reflector_S: callable = R_S,
        projector_M: callable = P_M,
    ) -> torch.Tensor:
    """
    Solvent Flipping algorithm iteration.

    Args:
        x: Complex input tensor [N,C,H,W].
        magnitude_S: Desired support magnitude [N,C,H,W].
        magnitude_M: Desired Fourier magnitude [N,C,H,W].
        support: Boolean support mask [N,C,H,W].
        reflector_S: Support reflector callable.
        projector_M: Magnitude projector callable.

    Returns:
        Updated tensor after SF iteration.
    """
    x_proj_M = projector_M(x, magnitude_M)
    x_reflect_S = reflector_S(x_proj_M, gamma_S=1.0, magnitude=magnitude_S, support=support)
    return x_reflect_S


def HIO(x, magnitude_S, magnitude_M, support, beta=0.7,
        projector_S=P_S, projector_M=P_M, enforce_magnitude=False):
    """HIO: projected field inside support, x-beta*projection outside.

    Default is support-only HIO. enforce_magnitude=True additionally imposes
    the measured object amplitude inside the support (a two-plane variant).
    """
    proj = projector_M(x, magnitude_M)
    inside = projector_S(proj, magnitude_S, support) if enforce_magnitude else proj
    return torch.where(support, inside, x - beta * proj)


def DM(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        gamma_S: float = None,
        gamma_M: float = None,
        projector_S: callable = P_S,
        projector_M: callable = P_M,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Difference Map (DM) algorithm iteration.
    (Random projections and the optimization of an algorithm for phase retrieval, J. Phys. A-Math Gen, Vol 36 pp 2995-3007)

    Args:
        x: Complex tensor of shape [N, C, H, W]
        magnitude_S: Support-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean support mask [N, C, H, W]
        beta: Relaxation parameter
        gamma_S: Parameter for support reflector (defaults to 1 / beta)
        gamma_M: Parameter for magnitude reflector (defaults to -1 / beta)
        projector_S: Callable for support projection
        projector_M: Callable for magnitude projection
        reflector_S: Callable for support reflection
        reflector_M: Callable for magnitude reflection

    Returns:
        Updated complex tensor after DM iteration
    """
    if gamma_S is None:
        gamma_S = 1.0 / beta
    if gamma_M is None:
        gamma_M = -1.0 / beta

    x_PMRS = projector_M(reflector_S(x, gamma_S, magnitude_S, support), magnitude_M)
    x_PSRM = projector_S(reflector_M(x, gamma_M, magnitude_M), magnitude_S, support)
    x_new = x + beta * (x_PMRS - x_PSRM)

    return x_new


def ASR(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Averaged Successive Reflections (ASR) algorithm iteration.

    Args:
        x: Complex tensor of shape [N, C, H, W]
        magnitude_S: Support-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean support mask [N, C, H, W]
        reflector_S: Callable for support reflection
        reflector_M: Callable for magnitude reflection

    Returns:
        Updated complex tensor after ASR iteration
    """
    r_m = reflector_M(x, gamma_M=1.0, magnitude=magnitude_M)
    r_s = reflector_S(r_m, gamma_S=1.0, magnitude=magnitude_S, support=support)
    return 0.5 * (r_s + x)


def HPR(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        projector_M: callable = P_M,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Hybrid Projection Reflection algorithm iteration (HPR).
    
    Args:
        x: Complex tensor [N, C, H, W]
        magnitude_S: Spatial-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean mask [N, C, H, W]
        beta: Relaxation parameter (default: 0.7)
        projector_M: Fourier magnitude projection function
        reflector_S: Support reflector
        reflector_M: Fourier magnitude reflector

    Returns:
        Updated complex tensor after one HPR iteration
    """
    proj = projector_M(x, magnitude_M)
    reflect_M = 2 * proj - x if reflector_M is R_M else reflector_M(x, gamma_M=1.0, magnitude=magnitude_M)
    x1 = reflector_S(reflect_M + (beta - 1) * proj, gamma_S=1.0, magnitude=magnitude_S, support=support)
    x2 = (1 - beta) * proj
    return 0.5 * (x1 + x + x2)


def RAAR(
        x: torch.Tensor,
        magnitude_S: torch.Tensor,
        magnitude_M: torch.Tensor,
        support: torch.Tensor,
        beta: float = 0.7,
        projector_M: callable = P_M,
        reflector_S: callable = R_S,
        reflector_M: callable = R_M,
    ) -> torch.Tensor:
    """
    Relaxed Averaged Alternating Reflectors (RAAR) algorithm iteration.
    
    Args:
        x: Complex tensor [N, C, H, W]
        magnitude_S: Spatial-domain magnitude [N, C, H, W]
        magnitude_M: Fourier-domain magnitude [N, C, H, W]
        support: Boolean mask [N, C, H, W]
        beta: Relaxation parameter (default: 0.7)
        projector_M: Fourier magnitude projector
        reflector_S: Spatial support reflector
        reflector_M: Fourier magnitude reflector

    Returns:
        Updated complex tensor after one RAAR iteration
    """
    proj = projector_M(x, magnitude_M)
    reflected = 2 * proj - x if reflector_M is R_M else reflector_M(x, gamma_M=1.0, magnitude=magnitude_M)
    asr = 0.5 * (reflector_S(reflected, 1., magnitude_S, support) + x)
    return beta * asr + (1 - beta) * proj


def smoothed_projection(x, magnitude_S, magnitude_M, support, beta=0.7,
                        sigma=1., filter_decay=0.99, iteration=0,
                        projector_S=P_S, projector_M=P_M):
    """Experimental globally smoothed projection heuristic; not published OSS."""
    proj = projector_S(projector_M(x, magnitude_M), magnitude_S, support)
    return x + beta * (supports.gaussian_smoothing(proj, sigma * filter_decay ** iteration) - x)


@lru_cache(maxsize=64)
def _oss_stages(shape, max_iter, n_stages):
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('OSS max_iter must be a positive integer')
    if not isinstance(n_stages, int) or n_stages < 1:
        raise ValueError('OSS n_stages must be a positive integer')
    h, w = shape
    count = min(n_stages, max_iter)
    length, remainder = divmod(max_iter, count)
    start, schedule = 0, []
    for stage in range(count):
        progress = stage / (count - 1) if count > 1 else 0.
        alpha = ((1-progress)*h + progress/h, (1-progress)*w + progress/w)
        stop = start + length + (stage < remainder)
        schedule.append((start, stop, alpha))
        start = stop
    return tuple(schedule)


def oss_bandwidth(shape, iteration=0, max_iter=2000, n_stages=10):
    """Staged Fourier-bin bandwidths, linearly decreasing from (H,W) to (1/H,1/W).

    Stages differ by at most one iteration; the first stages get the remainder.
    Short runs use min(n_stages,max_iter) nonempty stages. A single stage uses
    the initial broad filter. iteration is zero based and must be inside the run.
    """
    schedule = _oss_stages(tuple(shape), max_iter, n_stages)
    if not isinstance(iteration, int) or not 0 <= iteration < max_iter:
        raise ValueError('OSS iteration must satisfy 0 <= iteration < max_iter')
    return next(alpha for start, stop, alpha in schedule if start <= iteration < stop)


def OSS(x, magnitude_S, magnitude_M, support, beta=0.9, *, alpha=None,
        iteration=0, max_iter=2000, n_stages=10, positivity=False,
        enforce_magnitude=False, projector_S=P_S, projector_M=P_M):
    """One oversampling-smoothness update (Rodriguez et al., JAC 2013, eqs. 1-4).

    Apply Fourier magnitude projection, HIO, then Fourier Gaussian filtering of
    the ENTIRE HIO field; retain the filtered result only OUTSIDE the support.
    alpha is Fourier-bin bandwidth, not spatial sigma. run() supplies the stage
    schedule and carries each stage's lowest-R_F field into the next stage.

    Default: complex support-only extension for speckles. positivity=True uses
    the paper's real/nonnegative density constraint (discarding imaginary parts).
    enforce_magnitude=True is an optional two-plane extension for complex fields,
    not part of the original density algorithm. The two extensions are exclusive.
    Source: https://doi.org/10.1107/S0021889813002471
    """
    if not torch.is_complex(x) or x.ndim != 4:
        raise ValueError('OSS expects a complex NCHW field')
    if not 0 < beta <= 1:
        raise ValueError('OSS beta must be in (0,1]')
    if positivity and enforce_magnitude:
        raise ValueError('Choose positivity or measured magnitude, not both')
    if alpha is None:
        alpha = oss_bandwidth(x.shape[-2:], iteration, max_iter, n_stages)
    if positivity:
        proj = projector_M(x, magnitude_M).real
        accepted = support & (proj >= 0)
        density = torch.where(accepted, proj, x.real - beta * proj)
        hio = torch.complex(density, torch.zeros_like(density))
    else:
        hio = HIO(x, magnitude_S, magnitude_M, support, beta,
                  projector_S, projector_M, enforce_magnitude)
    filtered = transforms.fourier_gaussian_smoothing(hio, alpha)
    output = torch.where(support, hio, filtered)
    if positivity:
        output = torch.complex(output.real, torch.zeros_like(output.real))
    return output


def _is_oss(algorithm):
    return algorithm is OSS or isinstance(algorithm, partial) and _is_oss(algorithm.func)


def _apply_gaussian_smoothing(x, sigma):
    return supports.gaussian_smoothing(x, sigma)


@torch.no_grad()
def run(magnitudes, support=None, target_field=None, init=None, algorithm=None,
        algorithm_kwargs=None, max_iter=100, *, metric_every=1,
        track_best=False, return_best=False, selection_loss=None):
    """Run NCHW projection iterations with on-device metric histories.

    Metrics are recorded after each sampled update, including the final field.
    Optional best tracking compares Fourier amplitude MSE each iteration per
    batch/channel, including initialization. With return_best=True the last
    history entry is recomputed for the selected returned field. best_iteration
    uses zero for initialization. No reference field influences selection.

    OSS always carries the lowest scaled Fourier R-factor candidate of each
    stage into the next, independently per N,C, and returns the last stage's
    best by default. return_best=True explicitly overrides this with the global
    MSE choice, which is not the published OSS selection procedure.
    selection_loss(field) can override global best ranking with a per-N,C
    objective (e.g. detector-weighted likelihood). Reported Fourier MSE remains
    actual MSE; best_selection_loss records the ranking objective separately.
    """
    if algorithm is None:
        raise ValueError('You must provide an algorithm callable')
    if max_iter < 0 or metric_every < 1:
        raise ValueError('max_iter must be nonnegative; metric_every must be positive')
    near, far = (m.abs() for m in magnitudes)
    if near.ndim != 4 or near.shape != far.shape:
        raise ValueError('Expected matching NCHW magnitudes')
    kwargs = {} if algorithm_kwargs is None else dict(algorithm_kwargs)
    is_oss = _is_oss(algorithm)
    if is_oss:
        # Unwrap keyword partials so the runner and iteration share one schedule.
        while isinstance(algorithm, partial):
            if algorithm.args:
                raise ValueError('OSS partials must bind keyword options only')
            kwargs = {**algorithm.keywords, **kwargs}
            algorithm = algorithm.func
        if support is None:
            raise ValueError('OSS requires an explicit object support')
        schedule = _oss_stages(tuple(near.shape[-2:]), max_iter, kwargs.get('n_stages', 10)) if max_iter else ()
        kwargs['max_iter'] = max_iter
    support = torch.ones_like(near, dtype=torch.bool) if support is None else support
    x = initializers.flat_phases(near) if init is None else init.clone()
    if x.shape != near.shape:
        raise ValueError('Initialization shape must match magnitudes')
    x = transforms.apply_support(x, support)
    results = {key: [] for key in ('mse_fourier', 'nmse_fourier', 'mse_near',
                                   'pearson_fourier', 'pearson_fourier_intens',
                                   'r_factor_fourier')}
    if target_field is not None:
        results.update(quality=[], quality_phi=[])
    steps = []

    def record(field, spectrum, step):
        amplitude = spectrum.abs()
        results['mse_fourier'].append(metrics.mse(amplitude, far))
        results['nmse_fourier'].append(metrics.normalized_mse(amplitude, far))
        results['mse_near'].append(metrics.mse(field, near, mask=support))
        results['pearson_fourier'].append(metrics.pearson(amplitude, far, inversed=True))
        results['pearson_fourier_intens'].append(metrics.pearson(amplitude.square(), far.square(), inversed=True))
        results['r_factor_fourier'].append(metrics.fourier_r_factor(amplitude, far))
        if target_field is not None:
            results['quality'].append(metrics.quality(field, target_field, inversed=True, mask=support))
            results['quality_phi'].append(metrics.quality(
                torch.exp(1j * field.angle()), torch.exp(1j * target_field.angle()),
                inversed=True, mask=support))
        steps.append(step)

    track_best = track_best or return_best
    if track_best:
        best = x.clone()
        best_loss = (metrics.mse(transforms.fourier_transform(x), far)
                     if selection_loss is None else selection_loss(x))
        if best_loss.shape != near.shape[:2]:
            raise ValueError('selection_loss must return one value per N,C')
        best_step = torch.zeros_like(best_loss, dtype=torch.long)
    stage_index = 0
    stage_best = stage_loss = stage_step = None
    stage_scores, stage_steps = [], []
    for i in range(max_iter):
        if is_oss or algorithm is smoothed_projection:
            kwargs['iteration'] = i
        x = algorithm(x=x, magnitude_S=near, magnitude_M=far, support=support, **kwargs)
        sampled = (i + 1) % metric_every == 0 or i + 1 == max_iter
        if sampled or track_best or is_oss:
            spectrum = transforms.fourier_transform(x)
        if track_best:
            loss = metrics.mse(spectrum, far) if selection_loss is None else selection_loss(x)
            better = loss < best_loss
            best = torch.where(better[..., None, None], x, best)
            best_step = torch.where(better, i + 1, best_step)
            best_loss = torch.minimum(loss, best_loss)
        if is_oss:
            score = metrics.fourier_r_factor(spectrum, far)
            if stage_best is None:
                stage_best, stage_loss = x.clone(), score
                stage_step = torch.full_like(score, i + 1, dtype=torch.long)
            else:
                better = score < stage_loss
                stage_best = torch.where(better[..., None, None], x, stage_best)
                stage_step = torch.where(better, i + 1, stage_step)
                stage_loss = torch.minimum(score, stage_loss)
            if i + 1 == schedule[stage_index][1]:
                # A fresh stage starts from the previous stage's best candidate;
                # neither the stage input nor an earlier stage competes in it.
                x = stage_best
                spectrum = transforms.fourier_transform(x)
                stage_scores.append(stage_loss)
                stage_steps.append(stage_step)
                stage_best = stage_loss = stage_step = None
                stage_index += 1
        if sampled:
            record(x, spectrum, i + 1)
    if max_iter == 0:
        record(x, transforms.fourier_transform(x), 0)
    if return_best:
        x = best
        for values in results.values():
            values.pop()
        steps.pop()
        record(x, transforms.fourier_transform(x), max_iter)
    results = {key: torch.stack(values) for key, values in results.items()}
    results['iterations'] = torch.tensor(steps, device=x.device)
    if track_best:
        best_mse = best_loss if selection_loss is None else metrics.mse(transforms.fourier_transform(best), far)
        results.update(x_best=best, best_mse_fourier=best_mse,
                       best_selection_loss=best_loss, best_iteration=best_step)
    if is_oss:
        shape = (0, *near.shape[:2])
        results['oss_stage_best_r_factor'] = torch.stack(stage_scores) if stage_scores else near.new_empty(shape)
        results['oss_stage_best_iteration'] = torch.stack(stage_steps) if stage_steps else torch.empty(shape, device=near.device, dtype=torch.long)
        results['oss_stage_bounds'] = torch.tensor([(a, b) for a, b, _ in schedule], device=near.device, dtype=torch.long).reshape(-1, 2)
        alphas = [bandwidth if kwargs.get('alpha') is None else kwargs['alpha'] for _, _, bandwidth in schedule]
        alphas = [(a, a) if isinstance(a, (int, float)) else a for a in alphas]
        results['oss_stage_alpha'] = near.new_tensor(alphas).reshape(-1, 2)
    return x, results


def run_restarts(magnitudes, support=None, n_restarts=4, generator=None,
                 algorithm=RAAR, **run_kwargs):
    """Batch independent random-phase starts; select by residual per N,C.

    The restart axis is flattened into N during iteration. Returns selected
    fields and diagnostics containing all restart histories (N*R batch order),
    selected_restart [N,C], and selected Fourier amplitude MSE [N,C].
    For OSS, select final-stage candidates by scaled R_F instead, reported as
    r_factor_fourier; mse_fourier remains the selected candidate's actual MSE.
    """
    if n_restarts < 1:
        raise ValueError('n_restarts must be positive')
    near, far = magnitudes
    if near.ndim != 4 or near.shape != far.shape:
        raise ValueError('Expected matching NCHW magnitudes')
    n, c, h, w = near.shape

    def expand(tensor):
        tensor = torch.broadcast_to(tensor, near.shape)
        return tensor[:, None].expand(n, n_restarts, c, h, w).reshape(n*n_restarts, c, h, w)

    expanded = (expand(near), expand(far))
    mask = None if support is None else expand(support)
    kwargs = dict(run_kwargs)
    if 'init' in kwargs:
        raise ValueError('run_restarts generates its own random initialization')
    if kwargs.get('target_field') is not None:
        kwargs['target_field'] = expand(kwargs['target_field'])
    is_oss = _is_oss(algorithm)
    # OSS ranks final-stage candidates by R_F, not by an earlier stage's MSE.
    kwargs['return_best'] = not is_oss
    candidates, histories = run(expanded, support=mask, algorithm=algorithm,
                                init=initializers.random_phases(expanded[0], generator), **kwargs)
    if is_oss:
        candidate_spectrum = transforms.fourier_transform(candidates)
        losses = metrics.fourier_r_factor(candidate_spectrum, expanded[1]).reshape(n, n_restarts, c)
    elif kwargs.get('selection_loss') is not None:
        losses = histories['best_selection_loss'].reshape(n, n_restarts, c)
    else:
        losses = histories['best_mse_fourier'].reshape(n, n_restarts, c)
    selected_loss, selected = losses.min(dim=1)
    index = selected[:, None, :, None, None].expand(n, 1, c, h, w)
    output = candidates.reshape(n, n_restarts, c, h, w).gather(1, index).squeeze(1)
    diagnostics = {'selected_restart': selected, 'restart_results': histories}
    if is_oss:
        candidate_mse = metrics.mse(candidate_spectrum, expanded[1]).reshape(n, n_restarts, c)
        diagnostics['r_factor_fourier'] = selected_loss
        diagnostics['mse_fourier'] = candidate_mse.gather(1, selected[:, None]).squeeze(1)
    elif kwargs.get('selection_loss') is not None:
        diagnostics['selection_loss'] = selected_loss
        candidate_mse = histories['best_mse_fourier'].reshape(n, n_restarts, c)
        diagnostics['mse_fourier'] = candidate_mse.gather(1, selected[:, None]).squeeze(1)
    else:
        diagnostics['mse_fourier'] = selected_loss
    return output, diagnostics


def report_convergence_results(results: dict, yscale: str = 'log', key: str = 'pearson_fourier_intens'):
    """Format is [iteration, N, C]"""

    import matplotlib.pyplot as plt

    plt.figure()
    steps = results.get('iterations')
    steps = None if steps is None else steps.cpu()
    for i in range(results[key].shape[1]):
        values = results[key][:, i, 0].cpu()
        if steps is None:
            plt.plot(values)
        else:
            plt.plot(steps, values)
    plt.yscale(yscale)
    plt.title('Fourier intensity evolution (1 - PCC)')
    plt.ylabel('1 - PCC')
    plt.xlabel('Iteration')
    plt.grid(ls=':')
    # plt.legend()




def show_retrieved_fields(field, ft, x_hat, y_hat, window_crop: int = 60, power: float = 1, figsize: tuple[int, int] = (12, 7)):
    import numpy as np
    import matplotlib.pyplot as plt
    from speckleret.metrics import pearson
    from speckleret.plots import complex_to_hsv

    cropped = slice(window_crop, -window_crop) if window_crop >= 1 else slice(None)
    pcc_fourier_intensity = pearson(
        x=np.square(np.abs(ft[cropped, cropped])),
        y=np.square(np.abs(y_hat[cropped, cropped])),
    )

    hfig = plt.figure(figsize=figsize)

    plt.suptitle(f"Phase retrieval results: Fourier intensity PCC = {pcc_fourier_intensity*100:3.4f} %\n(Brightnesses = |field|^{power})")
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.power(np.abs(field[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Measured NF: ' + r'$|x|$' + f'^{power}')

    plt.subplot(2, 3, 2)
    plt.imshow(np.power(np.abs(x_hat[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Retrieved NF: ' + r'$|\hat{x}|$' + f'^{power}')

    plt.subplot(2, 3, 3)
    plt.title('Retrieved NF: ' + r'$\hat{x}$')
    plt.imshow(complex_to_hsv(np.abs(x_hat[cropped, cropped]) ** power * np.exp(1j * np.angle(x_hat[cropped, cropped])), rmin=0, rmax=np.max(np.power(np.abs(x_hat), power))))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(np.power(np.abs(ft[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Measured FF: ' + r'$|y|$' + f'^{power}')

    plt.subplot(2, 3, 5)
    plt.imshow(np.power(np.abs(y_hat[cropped, cropped]), power), cmap='gray')
    plt.axis('off')
    plt.title('Retrieved FF: ' + r'$|\hat{y}|$' + f'^{power}')

    plt.subplot(2, 3, 6)
    plt.imshow(complex_to_hsv(np.abs(y_hat[cropped, cropped]) ** power * np.exp(1j * np.angle(y_hat[cropped, cropped])), rmin=0, rmax=np.max(np.power(np.abs(y_hat), power))))
    plt.axis('off')
    plt.title('Retrieved FF: ' + r'$\hat{y}$')

    return hfig



"""Batched projection algorithms and measurement-based reconstruction tracking."""
import warnings
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


def OSS(*args, **kwargs):
    """Deprecated alias for smoothed_projection, which is not standard OSS."""
    warnings.warn('OSS implements a smoothing heuristic, not published OSS; '
                  'use smoothed_projection explicitly', DeprecationWarning, stacklevel=2)
    return smoothed_projection(*args, **kwargs)


def _apply_gaussian_smoothing(x, sigma):
    return supports.gaussian_smoothing(x, sigma)


@torch.no_grad()
def run(magnitudes, support=None, target_field=None, init=None, algorithm=None,
        algorithm_kwargs=None, max_iter=100, *, metric_every=1,
        track_best=False, return_best=False):
    """Run NCHW projection iterations with on-device metric histories.

    Metrics are recorded after each sampled update, including the final field.
    Optional best tracking compares Fourier amplitude MSE each iteration per
    batch/channel, including initialization. With return_best=True the last
    history entry is recomputed for the selected returned field. best_iteration
    uses zero for initialization. No reference field influences selection.
    """
    if algorithm is None:
        raise ValueError('You must provide an algorithm callable')
    if max_iter < 0 or metric_every < 1:
        raise ValueError('max_iter must be nonnegative; metric_every must be positive')
    near, far = (m.abs() for m in magnitudes)
    if near.ndim != 4 or near.shape != far.shape:
        raise ValueError('Expected matching NCHW magnitudes')
    kwargs = {} if algorithm_kwargs is None else dict(algorithm_kwargs)
    support = torch.ones_like(near, dtype=torch.bool) if support is None else support
    x = initializers.flat_phases(near) if init is None else init.clone()
    if x.shape != near.shape:
        raise ValueError('Initialization shape must match magnitudes')
    x = transforms.apply_support(x, support)
    results = {key: [] for key in ('mse_fourier', 'nmse_fourier', 'mse_near',
                                   'pearson_fourier', 'pearson_fourier_intens')}
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
        if target_field is not None:
            results['quality'].append(metrics.quality(field, target_field, inversed=True, mask=support))
            results['quality_phi'].append(metrics.quality(
                torch.exp(1j * field.angle()), torch.exp(1j * target_field.angle()),
                inversed=True, mask=support))
        steps.append(step)

    track_best = track_best or return_best
    if track_best:
        best = x.clone()
        best_loss = metrics.mse(transforms.fourier_transform(x), far)
        best_step = torch.zeros_like(best_loss, dtype=torch.long)
    for i in range(max_iter):
        if algorithm in (OSS, smoothed_projection):
            kwargs['iteration'] = i
        x = algorithm(x=x, magnitude_S=near, magnitude_M=far, support=support, **kwargs)
        sampled = (i + 1) % metric_every == 0 or i + 1 == max_iter
        if sampled or track_best:
            spectrum = transforms.fourier_transform(x)
        if track_best:
            loss = metrics.mse(spectrum, far)
            better = loss < best_loss
            best = torch.where(better[..., None, None], x, best)
            best_step = torch.where(better, i + 1, best_step)
            best_loss = torch.minimum(loss, best_loss)
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
        results.update(x_best=best, best_mse_fourier=best_loss, best_iteration=best_step)
    return x, results


def run_restarts(magnitudes, support=None, n_restarts=4, generator=None,
                 algorithm=RAAR, **run_kwargs):
    """Batch independent random-phase starts; select by residual per N,C.

    The restart axis is flattened into N during iteration. Returns selected
    fields and diagnostics containing all restart histories (N*R batch order),
    selected_restart [N,C], and selected Fourier amplitude MSE [N,C].
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
    kwargs['return_best'] = True
    candidates, histories = run(expanded, support=mask, algorithm=algorithm,
                                init=initializers.random_phases(expanded[0], generator), **kwargs)
    losses = histories['best_mse_fourier'].reshape(n, n_restarts, c)
    selected_loss, selected = losses.min(dim=1)
    index = selected[:, None, :, None, None].expand(n, 1, c, h, w)
    output = candidates.reshape(n, n_restarts, c, h, w).gather(1, index).squeeze(1)
    return output, {'selected_restart': selected, 'mse_fourier': selected_loss,
                    'restart_results': histories}


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



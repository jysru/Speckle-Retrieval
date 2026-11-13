import math

import numpy as np
import pytest
import torch

from speckleret.torch import initializers as init
from speckleret.torch import metrics, retrieval, supports, transforms as tf


@pytest.fixture(autouse=True, scope='module')
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field(shape=(2, 2, 7, 10), dtype=torch.complex128, device='cpu'):
    generator = torch.Generator(device=device).manual_seed(12)
    return torch.randn(shape, dtype=dtype, device=device, generator=generator)


@pytest.mark.parametrize('shape', [(2, 2, 7, 10), (1, 1, 8, 8), (1, 2, 9, 11)])
def test_centered_fft_matches_numpy_and_is_unitary(shape):
    x = field(shape)
    expected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x.numpy(), axes=(-2, -1)),
                                        norm='ortho'), axes=(-2, -1))
    actual = tf.fourier_transform(x)
    torch.testing.assert_close(actual, torch.from_numpy(expected))
    torch.testing.assert_close(tf.inverse_fourier_transform(actual), x)
    torch.testing.assert_close(actual.abs().square().sum((-2, -1)),
                               x.abs().square().sum((-2, -1)))


@pytest.mark.parametrize('target', [(7, 10), (5, 13), (10, 5), (4, 4), (12, 14)])
def test_resize_preserves_center_and_exact_shape(target):
    x = torch.zeros(1, 1, 7, 10)
    x[..., 3, 4] = 1
    actual = tf.resize_image(x, target)
    assert actual.shape[-2:] == target
    assert actual.sum() == 1
    # Compare against an independently constructed destination intersection.
    expected = torch.zeros(1, 1, *target)
    h, w = min(7, target[0]), min(10, target[1])
    expected[..., (target[0]-h)//2:(target[0]-h)//2+h,
             (target[1]-w)//2:(target[1]-w)//2+w] = x[..., (7-h)//2:(7-h)//2+h, (10-w)//2:(10-w)//2+w]
    torch.testing.assert_close(actual, expected)


def test_pearson_normalization_constants_and_shared_loss():
    x = torch.arange(4, dtype=torch.float64).reshape(1, 1, 2, 2)
    torch.testing.assert_close(metrics.pearson(x, x), torch.ones(1, 1, dtype=x.dtype))
    torch.testing.assert_close(metrics.pearson(x, -x), -torch.ones(1, 1, dtype=x.dtype))
    assert metrics.pearson(x, torch.ones_like(x)).item() == 0
    torch.testing.assert_close(init.batch_pearson_images(x, x, False, False),
                               metrics.pearson(x, x)[..., None, None])
    constant = torch.ones_like(x, requires_grad=True)
    init.PearsonImageLoss()(constant, x).backward()
    assert torch.isfinite(constant.grad).all()


def test_quality_detects_amplitude_errors_and_ignores_global_phase():
    x = torch.tensor([1., 100.], dtype=torch.float64).reshape(1, 1, 1, 2)
    y = x.flip(-1)
    assert metrics.quality(x, y).item() < 0.001
    assert metrics.weighted_phase_agreement(x, y).item() == 1
    phase = torch.exp(torch.tensor(1.3j, dtype=torch.complex128))
    torch.testing.assert_close(metrics.quality(x, 3 * x * phase), torch.ones(1, 1, dtype=x.dtype))
    assert metrics.quality(x, torch.zeros_like(x)).item() == 0


def test_normalized_mse_is_scale_invariant_and_handles_zero_target():
    x = field().abs()
    y = 1.5 * x
    expected = metrics.normalized_mse(x, y)
    torch.testing.assert_close(metrics.normalized_mse(x*1e-12, y*1e-12), expected)
    torch.testing.assert_close(metrics.normalized_mse(x, torch.zeros_like(y)), metrics.mse(x, torch.zeros_like(y)))


def test_masked_metrics_keep_batch_dimensions_and_ignore_outside_values():
    x = field()
    mask = torch.zeros(7, 10, dtype=torch.bool)
    mask[1:5, 2:8] = True
    y = torch.where(mask, x, 100 * x)
    q = metrics.quality(x, y, mask=mask)
    assert q.shape == (2, 2)
    torch.testing.assert_close(q, torch.ones_like(q))
    torch.testing.assert_close(metrics.mse(x, y, mask=mask), torch.zeros_like(q))
    for n in range(2):
        for c in range(2):
            torch.testing.assert_close(metrics.pearson(x, y, mask=mask)[n, c],
                                       metrics.pearson(x[n, c][mask][None, :], y[n, c][mask][None, :]))
    empty = torch.zeros_like(mask)
    assert torch.count_nonzero(metrics.quality(x, y, mask=empty)) == 0
    assert torch.count_nonzero(metrics.pearson(x, y, mask=empty)) == 0


@pytest.mark.parametrize('propagate', [tf.angular_spectrum_transform, tf.fresnel_propagation])
def test_propagation_zero_distance_and_plane_wave(propagate):
    x = field((1, 1, 9, 12))
    torch.testing.assert_close(propagate(x, dz=0), x)
    # A tilted plane wave has one spectral bin with a known longitudinal phase.
    spectrum = torch.zeros_like(x)
    spectrum[..., 4, 7] = 1
    wave = tf.inverse_fourier_transform(spectrum)
    wavelength, dx, dz = 1e-6, 5e-6, 7e-6
    k, kx = 2 * math.pi / wavelength, 2 * math.pi / (12 * dx)
    kz = math.sqrt(k*k-kx*kx) if propagate is tf.angular_spectrum_transform else k-kx*kx/(2*k)
    expected = wave * complex(math.cos(dz*kz), math.sin(dz*kz))
    torch.testing.assert_close(propagate(wave, dz=dz, wavelength=wavelength, pixel_size=dx), expected)


def test_propagation_inverse_cache_and_sampling():
    x = field()
    dy, dx = 4e-6, 7e-6
    grids = tf.fourier_grids(x, (dy, dx))
    assert all(grid.shape == x.shape[-2:] and grid.dtype == x.real.dtype for grid in grids)
    torch.testing.assert_close(grids[2][0], 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(10, dx, dtype=x.real.dtype)))
    forward = tf.angular_spectrum_transform(x, dz=1e-4, pixel_size=(dy, dx))
    back = tf.angular_spectrum_transform(forward, dz=-1e-4, pixel_size=(dy, dx))
    torch.testing.assert_close(back, x)
    a = tf.propagation_transfer_function(x, dz=1e-4)
    b = tf.propagation_transfer_function(x, dz=1e-4)
    assert a.data_ptr() == b.data_ptr()
    tf.clear_propagation_cache()
    assert tf.propagation_transfer_function(x, dz=1e-4).data_ptr() != a.data_ptr()


def test_evanescent_modes_decay_instead_of_acquiring_real_phase():
    x = field((1, 1, 8, 8))
    kwargs = dict(dz=1e-7, wavelength=1e-6, pixel_size=1e-7)
    discard = tf.propagation_transfer_function(x, **kwargs)
    decay = tf.propagation_transfer_function(x, **kwargs, evanescent='decay')
    assert discard[0, 0] == 0
    assert 0 < decay[0, 0].abs() < 1
    with pytest.raises(ValueError, match='Backward'):
        tf.propagation_transfer_function(x, dz=-1e-7, pixel_size=1e-7, evanescent='decay')


def test_legacy_propagator_phase_is_not_treated_as_transfer_function():
    x = field()
    phase = torch.full(x.shape[-2:], 0.3, dtype=x.real.dtype)
    with pytest.warns(DeprecationWarning):
        actual = tf.fresnel_transform(x, propagator=phase)
    torch.testing.assert_close(actual, x * torch.exp(0.3j * torch.ones_like(phase)))


def test_hio_feedback_and_explicit_magnitude_variant():
    x = field()
    support = supports.disk_support(x, radius=2)
    near, far = x.abs(), tf.fourier_transform(x).abs() * 1.3
    projection = retrieval.P_M(x, far)
    beta = 0.7
    actual = retrieval.HIO(x, near, far, support, beta=beta)
    torch.testing.assert_close(actual[~support], (x-beta*projection)[~support])
    torch.testing.assert_close(actual[support], projection[support])
    assert not torch.equal(actual[~support], x[~support])
    constrained = retrieval.HIO(x, near, far, support, enforce_magnitude=True)
    torch.testing.assert_close(constrained[support].abs(), near[support])


@pytest.mark.parametrize('algorithm', [retrieval.ER, retrieval.SF, retrieval.HIO,
                                      retrieval.DM, retrieval.ASR, retrieval.HPR,
                                      retrieval.RAAR, retrieval.smoothed_projection])
def test_algorithms_batch_independence(algorithm):
    x = field()
    support = supports.disk_support(x, 2)
    near, far = x.abs(), tf.fourier_transform(x).abs()
    batched = algorithm(x, near, far, support)
    separate = torch.cat([algorithm(x[i:i+1], near[i:i+1], far[i:i+1], support[i:i+1])
                          for i in range(x.shape[0])])
    torch.testing.assert_close(batched, separate)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_rectangular_support_and_shrinkwrap_independence(dtype):
    x = field(dtype=torch.complex128).abs().to(dtype)
    for support_fn in (supports.disk_support, supports.square_support):
        assert support_fn(x, 2).shape == x.shape
    batched = supports.shrinkwrap(x)
    separate = torch.cat([supports.shrinkwrap(x[i:i+1]) for i in range(2)])
    assert torch.equal(batched, separate)
    assert not supports.threshold_support(torch.zeros_like(x)).any()
    assert not supports.shrinkwrap(torch.zeros_like(x)).any()


def test_projection_idempotence():
    x = field()
    mask = supports.disk_support(x, 2)
    mag = x.abs() + 0.2
    projected = retrieval.P_S(x, mag, mask)
    torch.testing.assert_close(retrieval.P_S(projected, mag, mask), projected)
    projected = retrieval.P_M(x, mag)
    torch.testing.assert_close(retrieval.P_M(projected, mag), projected)


def test_runner_final_metrics_and_masked_quality():
    x = field()
    support = supports.disk_support(x, 2)
    near, far = x.abs(), tf.fourier_transform(x).abs()
    out, result = retrieval.run((near, far), support, target_field=x,
                                algorithm=retrieval.ER, max_iter=5, metric_every=2)
    assert result['iterations'].tolist() == [2, 4, 5]
    assert result['quality'].shape == (3, 2, 2)
    torch.testing.assert_close(result['mse_fourier'][-1], metrics.mse(tf.fourier_transform(out), far))
    torch.testing.assert_close(result['quality'][-1], metrics.quality(out, x, mask=support, inversed=True))


def test_best_selection_is_per_image_and_metrics_match_returned_field():
    near = torch.zeros(2, 1, 4, 4, dtype=torch.float64)
    far = torch.ones_like(near)
    calls = []

    def algorithm(**kwargs):
        amplitudes = ([1., 3.], [2., 1.])[len(calls)]
        calls.append(1)
        spectrum = torch.tensor(amplitudes, dtype=torch.float64)[:, None, None, None] * torch.ones_like(far)
        return tf.inverse_fourier_transform(spectrum)

    out, result = retrieval.run((near, far), algorithm=algorithm, max_iter=2, return_best=True)
    assert result['best_iteration'].tolist() == [[1], [2]]
    torch.testing.assert_close(tf.fourier_transform(out).abs(), far)
    torch.testing.assert_close(result['mse_fourier'][-1], result['best_mse_fourier'])


def test_zero_iterations_and_kwargs_not_mutated():
    x = field()
    mag = x.abs()
    kwargs = {'beta': 0.5}
    out, result = retrieval.run((mag, mag), algorithm=retrieval.ER, max_iter=0)
    torch.testing.assert_close(out.abs(), mag)
    assert result['iterations'].tolist() == [0]
    retrieval.run((mag, mag), algorithm=retrieval.smoothed_projection,
                  algorithm_kwargs=kwargs, max_iter=2)
    assert kwargs == {'beta': 0.5}


def test_cyclic_loss_enforces_fourier_magnitude_and_has_phase_gradient():
    x = field((1, 1, 5, 6))
    mag, target = x.abs(), tf.fourier_transform(x).abs()
    phase = torch.zeros_like(mag)
    model = init.PhaseOnlyOptimizer(mag, phase)
    spectrum = tf.fourier_transform(model())
    back = tf.inverse_fourier_transform(target * torch.exp(1j * spectrum.angle()))
    expected = 0.5 * (metrics.normalized_mse(spectrum, target) + metrics.normalized_mse(back, mag)).mean()
    actual = model.loss(target, mag)
    torch.testing.assert_close(actual, expected)
    g_plain = torch.autograd.grad(model.loss(target), model.phase)[0]
    g_cyclic = torch.autograd.grad(actual, model.phase)[0]
    assert (g_cyclic - 0.5*g_plain).abs().max() > 1e-6


@pytest.mark.parametrize('loss_type', ['amplitude_mse', 'intensity_mse', 'pearson_amplitude', 'pearson_intensity'])
def test_optimizer_returns_supported_field_with_final_loss(loss_type):
    x = field((1, 1, 7, 10))
    support = supports.disk_support(x, 2)
    far = tf.fourier_transform(tf.apply_support(x, support)).abs()
    out, history = init.gradient_descent(x.abs(), far, support, max_iter=3,
                                        return_loss=True, metric_every=2, loss_type=loss_type)
    assert out.dtype == x.dtype
    assert torch.count_nonzero(out[~support]) == 0
    assert len(history) == 3  # initialization, iteration 2, iteration 3
    check = init.PhaseOnlyOptimizer(x.abs(), out.angle(), support, loss_type)
    assert history[-1] == pytest.approx(check.loss(far).item())
    assert all(math.isfinite(value) for value in history)


def test_seeded_initialization_preserves_precision_and_batch_shapes():
    mag = field().abs()
    generator = torch.Generator().manual_seed(5)
    a = init.random_phases(mag, generator)
    b = init.random_phases(mag, torch.Generator().manual_seed(5))
    torch.testing.assert_close(a, b)
    assert a.dtype == torch.complex128
    assert init.flat_phases(mag).dtype == torch.complex128
    for fn in (init.thresholded_random_phase, init.autocorrelation_svd, init.random_phase_multistart):
        out = fn(mag, generator=torch.Generator().manual_seed(5))
        assert out.shape == mag.shape
        assert out.dtype == torch.complex128


def test_thresholded_initializer_drops_documented_fraction():
    mag = torch.arange(1, 17, dtype=torch.float64).reshape(1, 1, 4, 4)
    out = init.thresholded_random_phase(mag, drop_fraction=0.25,
                                       generator=torch.Generator().manual_seed(1))
    spectrum = tf.fourier_transform(out).abs()
    assert (spectrum < 1e-10).sum() == 4


def test_synthetic_phase_optimization_reduces_measurement_error():
    x = field((2, 1, 8, 10))
    support = supports.disk_support(x, 3)
    target = tf.apply_support(x, support)
    far = tf.fourier_transform(target).abs()
    out, history = init.gradient_descent(target.abs(), far, support, max_iter=120,
                                        initialization='random', generator=torch.Generator().manual_seed(4),
                                        return_loss=True, metric_every=20)
    # This nonconvex random-speckle case can stagnate; require a substantial
    # residual reduction, not guaranteed exact phase recovery.
    assert history[-1] < 0.2 * history[0]
    assert metrics.normalized_mse(tf.fourier_transform(out), far).max() < 0.1


def test_restarts_select_independently_and_preserve_diagnostics():
    x = field((2, 2, 7, 10))
    mask = supports.disk_support(x, 2)
    target = tf.apply_support(x, mask)
    near, far = target.abs(), tf.fourier_transform(target).abs()
    out, result = retrieval.run_restarts((near, far), mask, n_restarts=3,
                                        generator=torch.Generator().manual_seed(8),
                                        algorithm=retrieval.ER, max_iter=5,
                                        target_field=target)
    assert out.shape == target.shape
    assert result['selected_restart'].shape == (2, 2)
    assert result['restart_results']['quality'].shape == (5, 6, 2)
    all_losses = result['restart_results']['best_mse_fourier'].reshape(2, 3, 2)
    torch.testing.assert_close(result['mse_fourier'], all_losses.min(dim=1).values)
    torch.testing.assert_close(metrics.mse(tf.fourier_transform(out), far), result['mse_fourier'])


def test_raar_and_hpr_reuse_the_supplied_magnitude_projection():
    x = field()
    near, far = x.abs(), tf.fourier_transform(x).abs()
    mask = supports.disk_support(x, 2)
    calls = []

    def projector(value, magnitude):
        calls.append(1)
        return 0.8 * value

    beta = 0.6
    proj = 0.8 * x
    reflected = 2 * proj - x
    expected = beta * 0.5 * (retrieval.R_S(reflected, 1., near, mask) + x) + (1-beta) * proj
    actual = retrieval.RAAR(x, near, far, mask, beta=beta, projector_M=projector)
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    calls.clear()
    retrieval.HPR(x, near, far, mask, projector_M=projector)
    assert len(calls) == 1


def test_seeded_raar_reconstruction_recovers_reference_up_to_global_phase():
    x = field((2, 1, 16, 16), dtype=torch.complex64)
    mask = supports.disk_support(x, 4)
    target = tf.apply_support(x, mask)
    near, far = target.abs(), tf.fourier_transform(target).abs()
    out, _ = retrieval.run_restarts((near, far), mask, n_restarts=4,
                                    generator=torch.Generator().manual_seed(1000),
                                    max_iter=120, metric_every=20)
    assert metrics.normalized_mse(tf.fourier_transform(out), far).max() < 1e-5
    assert metrics.quality(out, target, mask=mask).min() > 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_matches_cpu_and_optimization_runs():
    x = field(dtype=torch.complex64)
    gpu = x.cuda()
    for operation in (tf.fourier_transform, tf.angular_spectrum_transform, supports.shrinkwrap):
        actual, expected = operation(gpu).cpu(), operation(x)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    far = tf.fourier_transform(gpu).abs()
    out = init.gradient_descent(gpu.abs(), far, max_iter=2)
    assert out.is_cuda

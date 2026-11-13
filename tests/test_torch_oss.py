"""OSS equations, schedule, and per-image stage selection regressions."""
from functools import partial

import numpy as np
import pytest
import torch

from speckleret.torch import metrics, retrieval, supports, transforms as tf


@pytest.fixture(autouse=True, scope='module')
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(dtype=torch.complex128):
    return torch.randn((2, 2, 5, 8), dtype=dtype,
                        generator=torch.Generator().manual_seed(23))


def numpy_fft(x, inverse=False):
    operation = np.fft.ifft2 if inverse else np.fft.fft2
    return np.fft.fftshift(operation(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'),
                           axes=(-2, -1))


@pytest.mark.parametrize('positivity,enforce_magnitude', [(False, False), (False, True), (True, False)])
def test_oss_matches_independent_numpy_equations(positivity, enforce_magnitude):
    x = sample()
    near = x.abs() + .2
    far = tf.fourier_transform(x).abs() * 1.2
    mask = supports.disk_support(x, 1.5)
    beta, alpha = .8, (1.3, 2.1)
    obj, mag, measured, support = (v.numpy() for v in (x, near, far, mask))
    projection = numpy_fft(measured * np.exp(1j * np.angle(numpy_fft(obj))), inverse=True)
    if positivity:
        good = support & (projection.real >= 0)
        hio = np.where(good, projection.real, obj.real - beta * projection.real)
    else:
        inside = mag * np.exp(1j * np.angle(projection)) if enforce_magnitude else projection
        hio = np.where(support, inside, obj - beta * projection)
    ky = np.arange(5) - 2
    kx = np.arange(8) - 4
    window = np.exp(-.5 * ((ky[:, None]/alpha[0])**2 + (kx[None, :]/alpha[1])**2))
    smooth = numpy_fft(numpy_fft(hio) * window, inverse=True)
    expected = np.where(support, hio, smooth)
    if positivity:
        expected = expected.real.astype(np.complex128)
    actual = retrieval.OSS(x, near, far, mask, beta=beta, alpha=alpha,
                           positivity=positivity, enforce_magnitude=enforce_magnitude)
    torch.testing.assert_close(actual, torch.from_numpy(expected))
    # Filtering uses the whole HIO field, not a zeroed interior.
    incorrectly_masked = numpy_fft(numpy_fft(np.where(support, 0, hio))*window, inverse=True)
    assert not np.allclose(expected[~support], incorrectly_masked[~support])


def test_filter_leaves_hio_interior_untouched_and_narrow_limit_keeps_dc():
    x = sample()
    near, far = x.abs(), tf.fourier_transform(x).abs() * 1.2
    mask = supports.disk_support(x, 1.5)
    hio = retrieval.HIO(x, near, far, mask, beta=.8)
    out = retrieval.OSS(x, near, far, mask, beta=.8, alpha=1e-6)
    torch.testing.assert_close(out[mask], hio[mask])
    mean = hio.mean((-2, -1), keepdim=True).expand_as(hio)
    torch.testing.assert_close(out[~mask], mean[~mask])
    assert torch.count_nonzero(out[~mask]) > 0
    assert not torch.allclose(out, retrieval.smoothed_projection(x, near, far, mask))


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_gaussian_attenuates_a_known_fourier_bin_and_preserves_dc(dtype):
    spectrum = torch.zeros((2, 1, 7, 10), dtype=dtype)
    spectrum[..., 3, 5] = 2
    spectrum[..., 4, 7] = 1
    out = tf.fourier_gaussian_smoothing(tf.inverse_fourier_transform(spectrum), (2., 3.))
    filtered = tf.fourier_transform(out)
    torch.testing.assert_close(filtered[..., 3, 5], spectrum[..., 3, 5])
    factor = np.exp(-.5 * ((1/2)**2 + (2/3)**2))
    torch.testing.assert_close(filtered[..., 4, 7], spectrum[..., 4, 7] * factor)
    assert out.dtype == dtype


def test_stage_schedule_covers_nondivisible_and_short_runs():
    # Seven updates: three broad, two intermediate, then two narrow.
    values = [retrieval.oss_bandwidth((5, 8), i, 7, 3) for i in range(7)]
    assert values[:3] == [(5., 8.)] * 3
    assert values[3:5] == [(2.6, 4.0625)] * 2
    assert values[5:] == [(1/5, 1/8)] * 2
    assert retrieval.oss_bandwidth((5, 8), 0, 1, 10) == (5., 8.)
    assert retrieval.oss_bandwidth((5, 8), 1, 2, 10) == (1/5, 1/8)
    assert retrieval.oss_bandwidth((16, 16), 199, 2000, 10) == (16., 16.)
    assert retrieval.oss_bandwidth((16, 16), 1999, 2000, 10) == (1/16, 1/16)


@pytest.mark.parametrize('options', [dict(beta=0), dict(beta=1.1), dict(alpha=0),
                                    dict(alpha=float('nan')), dict(alpha=(-1, 2)),
                                    dict(positivity=True, enforce_magnitude=True),
                                    dict(n_stages=0), dict(iteration=3, max_iter=3)])
def test_invalid_parameters_are_rejected(options):
    x = sample()
    with pytest.raises(ValueError):
        retrieval.OSS(x, x.abs(), x.abs(), torch.ones_like(x.real, dtype=torch.bool), **options)


def test_scaled_r_factor_has_defined_zero_cases_and_per_image_scaling():
    y = torch.tensor([1., 2., 3., 4.], dtype=torch.float64).reshape(1, 1, 2, 2)
    x = torch.cat([2*y, 3*y])
    torch.testing.assert_close(metrics.fourier_r_factor(x, y), torch.zeros(2, 1, dtype=y.dtype))
    assert metrics.fourier_r_factor(torch.zeros_like(y), y).item() == 1
    assert metrics.fourier_r_factor(y, torch.zeros_like(y)).item() == 10
    assert metrics.fourier_r_factor(torch.zeros_like(y), torch.zeros_like(y)).item() == 0
    perturbed = y.clone()
    perturbed[..., 0, 0] += 2
    gamma = (perturbed*y).sum()/perturbed.square().sum()
    expected = (y-gamma*perturbed).abs().sum()/y.sum()
    torch.testing.assert_close(metrics.fourier_r_factor(perturbed, y).squeeze(), expected)


def test_stage_handoff_uses_per_image_best_and_resets_ranking(monkeypatch):
    far = torch.ones(2, 1, 2, 2, dtype=torch.float64)
    near = torch.ones_like(far)
    # Each row is a Fourier amplitude; rankings differ across batch items.
    patterns = [
        [[1, 1, 1, 1], [1, 1, 1, 2]],
        [[1, 1, 1, 2], [1, 1, 1, 1]],
        [[1, 1, 2, 2], [1, 1, 3, 3]],
        [[1, 1, 4, 4], [1, 1, 2, 2]],
    ]
    candidates = [tf.inverse_fourier_transform(torch.tensor(p, dtype=torch.float64).reshape_as(far))
                  for p in patterns]
    inputs, options = [], []

    def fake_oss(x, **kwargs):
        inputs.append(x.clone())
        options.append(kwargs)
        return candidates[len(inputs)-1]

    monkeypatch.setattr(retrieval, 'OSS', fake_oss)
    kwargs = {'n_stages': 2}
    out, info = retrieval.run((near, far), torch.ones_like(near, dtype=torch.bool),
                              algorithm=fake_oss, algorithm_kwargs=kwargs, max_iter=4,
                              metric_every=3)
    torch.testing.assert_close(inputs[2], torch.cat([candidates[0][:1], candidates[1][1:]]))
    torch.testing.assert_close(out, torch.cat([candidates[2][:1], candidates[3][1:]]))
    assert info['oss_stage_best_iteration'].tolist() == [[[1], [2]], [[3], [4]]]
    assert info['oss_stage_bounds'].tolist() == [[0, 2], [2, 4]]
    assert info['iterations'].tolist() == [3, 4]
    torch.testing.assert_close(info['r_factor_fourier'][-1], info['oss_stage_best_r_factor'][-1])
    assert [option['iteration'] for option in options] == [0, 1, 2, 3]
    assert all(option['max_iter'] == 4 for option in options)
    assert kwargs == {'n_stages': 2}


def test_keyword_partial_and_zero_updates_work_with_the_runner():
    x = sample()
    mask = supports.disk_support(x, 1.5)
    mags = (x.abs(), tf.fourier_transform(x).abs())
    out, info = retrieval.run(mags, mask, init=x, algorithm=partial(retrieval.OSS, alpha=1.2, n_stages=2), max_iter=3)
    assert out.shape == x.shape
    torch.testing.assert_close(info['oss_stage_alpha'], x.real.new_full((2, 2), 1.2))
    out, info = retrieval.run(mags, mask, init=x, algorithm=retrieval.OSS, max_iter=0)
    torch.testing.assert_close(out, tf.apply_support(x, mask))
    assert info['oss_stage_bounds'].shape == (0, 2)
    assert info['oss_stage_best_r_factor'].shape == (0, 2, 2)
    with pytest.raises(ValueError, match='support'):
        retrieval.run(mags, algorithm=retrieval.OSS, max_iter=3)


def test_complex_runs_are_independent_per_batch_and_channel():
    x = sample()
    mask = supports.disk_support(x, 1.5)
    mags = (x.abs(), tf.fourier_transform(x).abs())
    kwargs = dict(algorithm=retrieval.OSS, max_iter=13,
                  algorithm_kwargs={'enforce_magnitude': True, 'n_stages': 3}, metric_every=4)
    out, info = retrieval.run(mags, mask, init=x, **kwargs)
    for n in range(2):
        for c in range(2):
            sl = (slice(n, n+1), slice(c, c+1))
            single, history = retrieval.run(tuple(m[sl] for m in mags), mask[sl], init=x[sl], **kwargs)
            torch.testing.assert_close(out[sl], single)
            torch.testing.assert_close(info['oss_stage_best_r_factor'][:, n, c], history['oss_stage_best_r_factor'][:, 0, 0])


def test_oss_restarts_rank_final_stage_by_r_factor_not_global_mse():
    x = sample()
    mask = supports.disk_support(x, 1.5)
    target = tf.apply_support(x, mask)
    near, far = target.abs(), tf.fourier_transform(target).abs()
    out, info = retrieval.run_restarts((near, far), mask, algorithm=retrieval.OSS,
                                      n_restarts=3, max_iter=20,
                                      generator=torch.Generator().manual_seed(18))
    stage_scores = info['restart_results']['oss_stage_best_r_factor'][-1].reshape(2, 3, 2)
    torch.testing.assert_close(info['r_factor_fourier'], stage_scores.min(dim=1).values)
    torch.testing.assert_close(info['r_factor_fourier'], metrics.fourier_r_factor(tf.fourier_transform(out), far))
    torch.testing.assert_close(info['mse_fourier'], metrics.mse(tf.fourier_transform(out), far))
    assert 'best_mse_fourier' not in info['restart_results']


def test_positive_density_with_poisson_noise_has_consistent_final_diagnostics():
    generator = torch.Generator().manual_seed(3)
    truth = torch.rand((2, 1, 16, 16), dtype=torch.float64, generator=generator)
    mask = supports.disk_support(truth, 4)
    truth = tf.apply_support(truth, supports.disk_support(truth, 3))
    intensity = tf.fourier_transform(truth).abs().square()
    peak = intensity.amax((-2, -1), keepdim=True)
    counts = torch.poisson(intensity / peak * 1000, generator=generator)
    far = (counts * peak / 1000).sqrt()
    # Placeholder initialization amplitude does not reveal the true density.
    out, info = retrieval.run_restarts((torch.ones_like(far), far), mask,
                                      algorithm=retrieval.OSS, max_iter=200,
                                      algorithm_kwargs={'positivity': True}, n_restarts=3,
                                      generator=generator, metric_every=25)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out.imag) == 0
    assert info['restart_results']['oss_stage_best_r_factor'].shape == (10, 6, 1)
    assert info['r_factor_fourier'].max() < .3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_oss_cuda_parity_for_single_step_and_staged_run():
    x = sample(torch.complex64)
    mask = supports.disk_support(x, 1.5)
    mags = (x.abs(), tf.fourier_transform(x).abs())
    cpu = retrieval.OSS(x, *mags, mask, alpha=(1.5, 2.))
    gpu = retrieval.OSS(x.cuda(), *(m.cuda() for m in mags), mask.cuda(), alpha=(1.5, 2.))
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=3e-4, atol=3e-5)
    out, info = retrieval.run(tuple(m.cuda() for m in mags), mask.cuda(), init=x.cuda(),
                              algorithm=retrieval.OSS, max_iter=20)
    assert out.is_cuda and info['oss_stage_alpha'].is_cuda

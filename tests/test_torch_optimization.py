import math

import pytest
import torch

from speckleret.torch import initializers, optimization as opt, retrieval
from speckleret.torch import supports, transforms as tf


@pytest.fixture(autouse=True, scope='module')
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(shape=(2, 2, 5, 8), dtype=torch.complex128):
    return torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(18))


def test_legacy_names_refer_to_new_module_and_lbfgs_runs():
    assert initializers.PhaseOnlyOptimizer is opt.PhaseOnlyOptimizer
    x = sample((1, 1, 5, 8))
    near, far = x.abs(), tf.fourier_transform(x).abs()
    out, history = initializers.gradient_descent(
        near, far, optimizer_class=torch.optim.LBFGS,
        optimizer_kwargs={'max_iter': 3}, max_iter=2, return_loss=True)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert len(history) == 3 and all(math.isfinite(loss) for loss in history)
    check = opt.PhaseOnlyOptimizer(near, out.angle())
    assert history[-1] == pytest.approx(check.loss(far).item())


def test_weighted_masked_loss_matches_hand_computation_and_ignores_nan_data():
    target = torch.tensor([1., 2., float('nan'), 4.], dtype=torch.float64).reshape(1, 1, 2, 2)
    mask = torch.tensor([True, True, False, True]).reshape_as(target)
    weight = torch.tensor([1., 2., float('nan'), 3.], dtype=target.dtype).reshape_as(target)
    field = torch.tensor([2., 1., 9., 3.], dtype=target.dtype).reshape_as(target).requires_grad_()
    term = opt.MeasurementLoss(target, mask=mask, weights=weight)
    expected = (1*(2-1)**2 + 2*(1-2)**2 + 3*(3-4)**2) / (1*1**2 + 2*2**2 + 3*4**2)
    assert term(field).item() == pytest.approx(expected)
    term(field).sum().backward()
    assert field.grad.flatten()[2] == 0
    assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize('loss_type', opt.LOSS_TYPES)
def test_loss_gradients_are_finite_and_independent_per_image(loss_type):
    x = sample()
    target = x.abs().square() + .3 if loss_type == 'poisson' else x.abs() + .3
    kind = 'counts' if loss_type == 'poisson' else 'amplitude'
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 0, 0] = False
    actual = opt.MeasurementLoss(target, input_type=kind, loss_type=loss_type, mask=mask)(x)
    assert actual.shape == (2, 2)
    for n in range(2):
        sl = slice(n, n+1)
        expected = opt.MeasurementLoss(target[sl], input_type=kind, loss_type=loss_type, mask=mask[sl])(x[sl])
        torch.testing.assert_close(actual[sl], expected)
    variable = x.clone().requires_grad_()
    opt.MeasurementLoss(target, input_type=kind, loss_type=loss_type, mask=mask)(variable).sum().backward()
    assert torch.isfinite(variable.grad).all()


def test_poisson_detector_model_matches_likelihood_and_gradcheck():
    x = sample((1, 1, 2, 3))
    counts = torch.tensor([0., 3., 5., 1., 0., 2.], dtype=torch.float64).reshape(1, 1, 2, 3)
    gain, background = 2.5, .4
    term = opt.MeasurementLoss(counts, input_type='counts', loss_type='poisson', gain=gain, background=background)
    rate = gain*x.abs().square()+background
    expected = (rate-counts + torch.xlogy(counts, counts.clamp_min(1e-300)) - counts*rate.log()).mean((-2, -1))
    torch.testing.assert_close(term(x), expected)
    assert torch.autograd.gradcheck(term, (x.clone().requires_grad_(),))
    zero = opt.MeasurementLoss(torch.zeros_like(counts), input_type='counts', loss_type='poisson')
    value = torch.zeros_like(x, requires_grad=True)
    zero(value).sum().backward()
    assert torch.isfinite(value.grad).all()


def test_calibrated_initial_amplitude_uses_gain_and_background():
    amplitude = sample().abs()
    intensity = 3*amplitude.square()+2
    term = opt.MeasurementLoss(intensity, input_type='intensity', gain=3., background=2.)
    torch.testing.assert_close(term.initial_amplitude(), amplitude)
    assert term(amplitude).max() < 1e-25
    # Integer photon counts are accepted and converted to floating-point data.
    out, info = opt.reconstruct((torch.ones(1, 1, 3, 3, dtype=torch.int64),)*2,
                                input_type='counts', loss_type='poisson', max_iter=1)
    assert out.dtype == torch.complex64 and torch.isfinite(info['final_loss']).all()


@pytest.mark.parametrize('options', [dict(loss_type='poisson'), dict(gain=0),
                                    dict(background=-1), dict(weights=-1),
                                    dict(huber_delta=0), dict(loss_type='typo')])
def test_invalid_measurement_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        opt.MeasurementLoss(torch.ones(1, 1, 3, 3), **options)


def test_observed_nan_negative_and_complex_measurements_are_rejected():
    for target in (torch.full((1, 1, 2, 2), float('nan')),
                   -torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.complex64)):
        with pytest.raises(ValueError):
            opt.reconstruct((target, target), max_iter=1)
    with pytest.raises(ValueError, match='observed'):
        opt.reconstruct((torch.ones(1, 1, 2, 2),)*2, masks=(None, torch.zeros(2, 2, dtype=torch.bool)))


def test_soft_amplitude_optimization_and_lbfgs_find_known_scalar_minimum():
    near = torch.ones(1, 1, 3, 4, dtype=torch.float64)
    far = 3*near
    fixed, fixed_info = opt.reconstruct((near, far), forward_model=lambda x: x, max_iter=2)
    fitted, info = opt.reconstruct((near, far), forward_model=lambda x: x,
                                   optimize_amplitude=True, optimizer_class=torch.optim.LBFGS,
                                   max_iter=4)
    # Minimize ((a-1)^2 + (a-3)^2/9)/2: optimum a=1.2, not a hard constraint.
    torch.testing.assert_close(fitted.abs(), torch.full_like(near, 1.2), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(fixed.abs(), near)
    assert info['final_loss'].item() < fixed_info['final_loss'].item()
    assert info['function_evaluations'] > info['iterations_completed']+1
    assert info['loss'][-1].item() == pytest.approx(info['final_loss'].item())


def test_first_order_sgd_is_batch_independent_including_updates():
    x = sample()
    near, far = x.abs(), tf.fourier_transform(x).abs()
    phase = .2*x.angle()
    options = dict(init_phase=phase, optimizer_class=torch.optim.SGD,
                   optimizer_kwargs={'lr': .03}, max_iter=4, return_best=False)
    batched, _ = opt.reconstruct((near, far), **options)
    for n in range(2):
        for c in range(2):
            sl = (slice(n, n+1), slice(c, c+1))
            single, _ = opt.reconstruct((near[sl], far[sl]), init_phase=phase[sl],
                                        optimizer_class=torch.optim.SGD, optimizer_kwargs={'lr': .03},
                                        max_iter=4, return_best=False)
            torch.testing.assert_close(batched[sl], single)


def test_custom_forward_operator_with_different_detector_shape():
    x = sample((1, 1, 5, 8))
    phase_factor = torch.exp(1j*torch.linspace(0., .4, 8, dtype=x.real.dtype))
    def forward(field):
        return tf.fourier_transform(field*phase_factor)[..., 1:4, 1:7]
    near, far = x.abs(), forward(x).abs()
    out, info = opt.reconstruct((near, far), forward_model=forward, max_iter=6)
    assert out.shape == x.shape
    assert info['final_loss'].item() < info['initial_loss'].item()
    term = opt.MeasurementLoss(far)
    torch.testing.assert_close(info['final_loss'], term(forward(out)))
    with pytest.raises(ValueError, match='inverse_model'):
        opt.reconstruct((near, far), forward_model=forward, cyclic_loss=True)


def test_per_image_best_tracking_matches_returned_field_and_sampling():
    near = sample().abs()
    far = 2*near
    out, info = opt.reconstruct((near, far), forward_model=lambda x: x,
                                optimize_amplitude=True, plane_weights=(0., 1.),
                                optimizer_class=torch.optim.SGD, optimizer_kwargs={'lr': 30.},
                                max_iter=7, metric_every=3)
    assert info['iterations'].tolist() == [0, 3, 6, 7]
    assert info['loss'].shape == (4, 2, 2)
    assert (info['final_loss'] <= info['initial_loss']).all()
    torch.testing.assert_close(info['final_loss'], opt.MeasurementLoss(far)(out))
    assert info['best_iteration'].shape == (2, 2)


def test_early_stopping_and_zero_updates_have_consistent_diagnostics():
    x = sample().abs()
    out, info = opt.reconstruct((x, x), forward_model=lambda field: field,
                                rel_tol=1e-8, patience=2, check_every=2, max_iter=30,
                                metric_every=3)
    assert info['iterations_completed'] == 4
    assert info['iterations'].tolist() == [0, 3, 4]
    assert info['stalled'].all()
    out, info = opt.reconstruct((x, x), max_iter=0)
    torch.testing.assert_close(out.abs(), x)
    assert info['iterations'].tolist() == [0]


def test_best_fields_can_come_from_different_updates_in_one_batch():
    near = torch.ones(2, 1, 1, 1, dtype=torch.float64)
    far = torch.tensor([1.1, 3.], dtype=torch.float64).reshape_as(near)
    out, info = opt.reconstruct((near, far), forward_model=lambda x: x,
                                optimize_amplitude=True, plane_weights=(0., 1.),
                                optimizer_class=torch.optim.SGD, optimizer_kwargs={'lr': 2.},
                                max_iter=5)
    assert info['best_iteration'].tolist() == [[0], [5]]
    assert out[0].abs().item() == 1
    torch.testing.assert_close(info['final_loss'], opt.MeasurementLoss(far)(out))


def test_one_stalled_image_does_not_stop_an_improving_batch():
    near = torch.ones(2, 1, 2, 2, dtype=torch.float64)
    far = near.clone()
    far[1] = 3
    _, info = opt.reconstruct((near, far), forward_model=lambda x: x,
                              optimize_amplitude=True, plane_weights=(0., 1.),
                              optimizer_class=torch.optim.SGD, optimizer_kwargs={'lr': .1},
                              rel_tol=1e-10, patience=1, check_every=2, max_iter=8)
    assert info['iterations_completed'] == 8
    assert info['stalled'].tolist() == [[True], [False]]


def test_zero_near_amplitude_does_not_trap_soft_field_at_zero():
    near = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    far = torch.ones_like(near)
    out, _ = opt.reconstruct((near, far), forward_model=lambda x: x,
                             optimize_amplitude=True, optimizer_class=torch.optim.LBFGS,
                             max_iter=3)
    # Both equally weighted amplitude losses have their minimum at amplitude .5.
    torch.testing.assert_close(out.abs(), torch.full_like(near, .5), rtol=1e-4, atol=1e-5)


def test_operator_modules_must_be_fixed_and_are_not_implicitly_optimized():
    operator = torch.nn.Conv2d(1, 1, 1)
    with pytest.raises(ValueError, match='Freeze operator'):
        opt.reconstruct((torch.ones(1, 1, 3, 3),)*2, forward_model=operator)


def test_explicit_optimizer_kwargs_preserve_optimizer_defaults():
    seen = []
    class CaptureAdam(torch.optim.Adam):
        def __init__(self, parameters, **kwargs):
            seen.append(kwargs)
            super().__init__(parameters, **kwargs)
    measurements = (torch.ones(1, 1, 3, 3),)*2
    opt.reconstruct(measurements, optimizer_class=CaptureAdam, optimizer_kwargs={}, max_iter=0)
    opt.reconstruct(measurements, optimizer_class=CaptureAdam, max_iter=0)
    assert seen == [{}, {'lr': .1}]


@pytest.mark.parametrize('loss_type', opt.LOSS_TYPES)
def test_nonfinite_initial_forward_model_fails_before_optimization(loss_type):
    with pytest.raises(ValueError, match='Initial reconstruction objective'):
        opt.reconstruct((torch.ones(1, 1, 3, 3),)*2,
                         forward_model=lambda field: field*float('nan'), max_iter=1,
                         input_type='counts' if loss_type == 'poisson' else 'amplitude',
                         loss_type=loss_type)


def test_custom_projection_selection_does_not_relabel_actual_fourier_mse():
    near = torch.ones(2, 1, 2, 2)
    def scorer(field):
        return (field.real-2).square().mean((-2, -1))
    def update(x, **kwargs):
        return x + 1
    out, info = retrieval.run((near, near), algorithm=update, max_iter=3,
                              return_best=True, selection_loss=scorer)
    torch.testing.assert_close(out.real, 2*near)
    torch.testing.assert_close(info['best_selection_loss'], torch.zeros(2, 1))
    torch.testing.assert_close(info['best_mse_fourier'], info['mse_fourier'][-1])
    assert info['best_mse_fourier'].min() > 0


def test_hybrid_selection_and_refinement_use_same_masked_weighted_objective():
    x = sample((2, 1, 8, 10))
    support = supports.disk_support(x, 2)
    x = tf.apply_support(x, support)
    near, far = x.abs(), tf.fourier_transform(x).abs()
    mask = torch.ones_like(far, dtype=torch.bool)
    mask[..., 0, 0] = False
    corrupted = far.clone()
    corrupted[..., 0, 0] = float('nan')
    out, info = opt.hybrid_reconstruct((near, corrupted), support, n_restarts=3,
                                       projection_iter=15, refinement_iter=4,
                                       masks=(None, mask), weights=(None, far+.2),
                                       generator=torch.Generator().manual_seed(7))
    assert out.shape == x.shape
    assert not out[~support].any()
    torch.testing.assert_close(info['projection_loss'], info['refinement']['initial_loss'])
    assert (info['refinement']['final_loss'] <= info['projection_loss']+1e-12).all()
    term = opt.MeasurementLoss(corrupted, mask=mask, weights=far+.2)
    torch.testing.assert_close(info['refinement']['final_loss'], term(tf.fourier_transform(out)))


def test_hybrid_counts_with_soft_amplitude_and_lbfgs_refinement():
    x = sample((1, 1, 6, 8))
    support = supports.disk_support(x, 2)
    x = tf.apply_support(x, support)
    gains, backgrounds = (2., 3.), (.1, .2)
    counts = (gains[0]*x.abs().square()+backgrounds[0],
              gains[1]*tf.fourier_transform(x).abs().square()+backgrounds[1])
    out, info = opt.hybrid_reconstruct(counts, support, input_type='counts', loss_type='poisson',
                                       gains=gains, backgrounds=backgrounds, optimize_amplitude=True,
                                       n_restarts=2, projection_iter=10, refinement_iter=2,
                                       optimizer_class=torch.optim.LBFGS,
                                       generator=torch.Generator().manual_seed(3))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(info['projection_loss'], info['refinement']['initial_loss'])
    assert (info['refinement']['final_loss'] <= info['projection_loss']).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_adam_and_lbfgs_keep_data_and_diagnostics_on_device():
    x = sample((1, 1, 5, 8), torch.complex64).cuda()
    for optimizer in (torch.optim.Adam, torch.optim.LBFGS):
        out, info = opt.reconstruct((x.abs(), tf.fourier_transform(x).abs()),
                                    optimizer_class=optimizer, max_iter=2)
        assert out.is_cuda and info['loss'].is_cuda
        assert torch.isfinite(info['final_loss']).all()

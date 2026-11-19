# Speckle-Retrieval

Phase retrieval of complex optical fields from near-field and far-field
measurements. The PyTorch implementation is the reference backend for ongoing
development; the NumPy implementation and older notebooks are retained for
research history and are not covered by the new correctness suite.

## Installation

Python 3.10 or newer is required. Install in a virtual environment:

```bash
python -m pip install -e .
# Include regression tests:
python -m pip install -e ".[dev]"
```

PyTorch is a declared dependency. For a particular CUDA build, install the
appropriate PyTorch wheel for your machine before installing this package.
Neither waveoptics nor torchmetrics is required. The existing
`python -m pip install -r requirements.txt` command installs the development
configuration from the repository root.

## PyTorch quick start

For an executed end-to-end example with the bundled GRIN speckle field, see
[the PyTorch reconstruction walkthrough](notebooks/pytorch_reconstruction_walkthrough.ipynb).
It compares Adam, RAAR followed by ER, and RAAR-ER followed by Adam on a shared
starting field, with side-by-side recovered phases and phase errors.

For experimental hollow-core camera data, see
[the experimental PyTorch reconstruction notebook](notebooks/experimental_hollow_core_pytorch_reconstruction.ipynb).
It loads the dated NF/FF NPZ pair, offers support and reconstruction choices,
and compares Fourier intensity PCC without modal decomposition.

Inputs have shape `[N,C,H,W]`: batch, channel, height, width. Magnitudes are
**field amplitudes**, not camera intensities. After background correction, a
nonnegative measured intensity is converted with `intensity.clamp_min(0).sqrt()`.
All arrays used together must be on the same device, with compatible precision.
Float32 amplitudes produce complex64 fields; float64 produces complex128.

This synthetic example requires no external data:

```python
import torch
from speckleret.torch import transforms, supports, retrieval, metrics

# Explicit CPU fallback also makes this example usable without CUDA.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = torch.Generator(device=device).manual_seed(7)
reference = torch.randn(2, 1, 32, 32, dtype=torch.complex64,
                        device=device, generator=generator)
support = supports.disk_support(reference, radius=8)
reference = transforms.apply_support(reference, support)
near = reference.abs()
far = transforms.fourier_transform(reference).abs()

field, diagnostics = retrieval.run_restarts(
    (near, far), support=support, n_restarts=4,
    generator=torch.Generator(device=device).manual_seed(8),
    algorithm=retrieval.RAAR, max_iter=200, metric_every=20,
)
print('Far-field NMSE:', metrics.normalized_mse(
    transforms.fourier_transform(field), far).cpu())
print('Reference overlap:', metrics.quality(field, reference, mask=support).cpu())
```

For direct phase-only optimization:

```python
from speckleret.torch import initializers

field, losses = initializers.gradient_descent(
    near, far, support,
    initialization='random',
    generator=torch.Generator(device=device).manual_seed(9),
    optimizer_kwargs={'lr': 0.1}, max_iter=300,
    loss_type='amplitude_mse', return_loss=True, metric_every=10,
)
```

Phase-only optimization fixes the near-field amplitude inside the support and
optimizes only phase. It can stagnate at local minima; try multiple starts and
compare residuals. No algorithm here guarantees unique recovery of arbitrary
fields from two intensity measurements.

## Automatic-differentiation reconstruction

`speckleret.torch.optimization` contains the reconstruction API.
`initializers.gradient_descent` and `initializers.PhaseOnlyOptimizer` remain
compatible aliases. `optimization.reconstruct()` returns `(field, diagnostics)`;
the older wrapper still returns a field or `(field, scalar_loss_list)`.

A useful workflow is several RAAR starts followed by AD refinement:

```python
from speckleret.torch import optimization

field, diagnostics = optimization.hybrid_reconstruct(
    (near, far), support,
    n_restarts=4, projection_iter=100, refinement_iter=20,
    optimizer_class=torch.optim.LBFGS,
    optimizer_kwargs={'max_iter': 5, 'history_size': 10,
                      'line_search_fn': 'strong_wolfe'},
    generator=torch.Generator(device=device).manual_seed(11),
)
print('Before AD:', diagnostics['refinement']['initial_loss'].cpu())
print('After AD:', diagnostics['refinement']['final_loss'].cpu())
```

Projection candidates are scored with the same detector objective used during
refinement, including masks, confidence weights, gains, and backgrounds. Each
image/channel selects its own candidate. Refinement returns its per-image best
by default, including initialization, so its objective cannot worsen. This does
not guarantee better reference-field accuracy or unique recovery. Hybrid
initialization currently requires equal-size Fourier-related planes; for a
custom forward model, call `reconstruct(init=...)` directly.
Phase-to-field conversion can introduce rounding differences between the raw
`projection_loss` and `refinement.initial_loss`, especially near numerical zero.

The AD parameterizations are:

- Phase-only (default): fixed calibrated near-field amplitude and learned phase.
- `optimize_amplitude=True`: learned real and imaginary field components, with
  soft near- and far-field data fidelity. No smooth-phase prior is imposed.
  Both planes have equal objective weight by default; `plane_weights=(w_near,
  w_far)` adjusts their influence. Phase-only fitting defaults to `(0,1)`.

If amplitude fitting has no explicit initial field and its near-plane guess is
entirely zero, far-plane energy provides a nonzero seed. This avoids the zero
stationary point of intensity objectives; for nonunitary custom models this is
only an initialization heuristic. Explicit initial fields are respected.

`support` is the object mask. Separate `masks=(near_mask, far_mask)` are boolean
**detector** masks, broadcastable to their observations; masked NaNs are ignored.
`weights=(near_weights, far_weights)` supplies finite nonnegative confidence
weights, with zero weight equivalent to a missing pixel. Each plane's loss is a
weighted spatial mean, then the plane weights are normalized by their sum.
Partially missing near amplitudes are initialized from that image's valid mean;
if they should be inferred rather than fixed to a guess, enable amplitude fitting.
Every image/channel must have observations in at least one weighted plane.

Available `loss_type` values are normalized `amplitude_mse`, `intensity_mse`,
`huber_amplitude`, `huber_intensity`, `pearson_amplitude`, `pearson_intensity`, and
`poisson`. Huber residuals are normalized by the target RMS and use a configurable
`huber_delta` (default 1). Pearson uses the same population-correlation convention
as the metrics module, extended to confidence weights. Constant-image correlation
is defined as zero; a completely unobserved plane contributes zero loss.

`input_type='amplitude'` is the default. `input_type='intensity'` accepts measured
sensor intensities. `input_type='counts'` is required for Poisson fitting. All
three interfaces use the explicit detector model
`expected_intensity = gain * abs(propagated_field)**2 + background`, with
`gains=(g_near,g_far)` and `backgrounds=(b_near,b_far)` broadcastable to each
plane. Defaults are unit gain and zero background. Corrections are applied only
for initialization; the likelihood still fits the original measurements.

For example, simulate count data and fit both amplitude and phase:

```python
counts = tuple(torch.poisson(100 * a.square()) for a in (near, far))
field, diagnostics = optimization.reconstruct(
    counts, support,
    input_type='counts', loss_type='poisson',
    gains=(100., 100.), backgrounds=(0., 0.),
    optimize_amplitude=True, max_iter=200,
    initialization='random',
    generator=torch.Generator(device=device).manual_seed(12),
)
```

The Poisson objective is mean weighted generalized KL divergence, equivalent to
Poisson negative log likelihood after subtracting a data-only constant. Zero
counts are handled explicitly. Do not label normalized, dark-subtracted amplitude
images as raw counts. Choose a Gaussian/robust loss for data that do not follow
the count model; arbitrary Gaussian noise added to counts is not pure Poisson.

`forward_model(field)` can provide any deterministic differentiable propagation
with NCHW output, including a different detector height/width, provided N,C and
measurement dimensions match. It must act independently on images/channels for
per-image selection to be meaningful. Only field parameters are optimized;
operator modules must have their parameters frozen. Joint calibration fitting is
not enabled implicitly. In particular, a free far-field shift is confounded with
an object-plane phase ramp. Legacy cyclic loss requires a matching
`inverse_model` for custom propagation and currently only supports complete,
unweighted, calibrated phase-only amplitude data.

Both Adam and L-BFGS have explicit support. Other ordinary first-order optimizer
classes work through `optimizer_class`. L-BFGS uses a closure and strong-Wolfe
line search by default. One outer update can evaluate the model many times, and
stock PyTorch L-BFGS shares one line search and curvature history across the
batch; it is not an independently vectorized optimizer. First-order objectives
are summed across images/channels so SGD updates do not change with batch size.
No default phase weight decay is introduced.

Diagnostics stay on the input device: `loss` has shape `[samples,N,C]`,
`iterations` identifies sampled updates, and `initial_loss`, `final_loss`,
`best_loss`, and `best_iteration` are per N,C. Histories include initialization
and the returned field. `iterations_completed` counts outer optimizer updates;
`function_evaluations` counts objective evaluations, including L-BFGS closures
and diagnostics. Histories are objective values, not necessarily Fourier NMSE.
`return_best=False` returns the last iterate instead of combining per-image best
fields. Optional `rel_tol`, `patience`, and `check_every` stop only when **all**
images stall; those periodic checks synchronize the device. Fixed iteration
counts remain the default. Logging and schedulers remain opt-in.

The hybrid diagnostics contain the selected restart, projection objective,
projection histories, and nested refinement diagnostics. The projection runner's
new `selection_loss(field)` hook allows custom per-image candidate ranking while
keeping its reported Fourier MSE separate from the selection objective.

## Losses, histories, and selection

- `amplitude_mse` (default): mean per-image amplitude MSE divided by mean target
  amplitude squared. `intensity_mse` compares squared amplitudes instead.
- `pearson_amplitude` and `pearson_intensity` are available explicitly; correlation
  alone does not penalize positive scaling or additive offsets.
- `gradient_descent(..., return_loss=True)` returns a Python list of sampled
  scalar losses, including initialization (step 0) and the final returned field.
  With `metric_every=k`, steps are `0,k,2k,...,max_iter`. Loss history is transferred
  from the device once; training forward passes are reused for intermediate losses.
- `retrieval.run()` histories describe sampled **post-update** fields and always
  include the last update. `results['iterations']` gives their step numbers.
  With zero updates, only initialization is evaluated.
- `track_best=True` tracks the lowest Fourier amplitude MSE independently per
  batch item/channel, including initialization. `return_best=True` also returns
  those fields and recomputes the last metrics for them. Inspect `best_iteration`
  to find their actual steps. Both-plane residuals should still be checked: a low
  Fourier residual alone does not establish a correct object field.
- `run_restarts()` flattens batch and restart axes internally and selects each
  image/channel independently. Its diagnostics include `selected_restart`, selected
  Fourier `mse_fourier`, and `restart_results` containing all histories in N*R order.

`metrics.quality` is squared L2-normalized complex overlap, invariant to global
phase and scale. The previous amplitude-weighted phase agreement is available as
`metrics.weighted_phase_agreement`. Metrics reduce H,W only, including when a
broadcastable boolean mask is supplied. Pearson returns zero for constant/empty
images; overlap returns zero for zero-energy inputs. These are explicit policies
for undefined cases, not evidence of physical disagreement.

## Oversampling smoothness (OSS)

`retrieval.OSS` implements the HIO update and Fourier Gaussian filtering of
[Rodriguez et al. (2013), equations 1-5](https://doi.org/10.1107/S0021889813002471).
The full HIO field is filtered; only values **outside** the support are replaced
by the filtered result. Inside-support HIO values remain unchanged by smoothing.
The previous global spatial-blur heuristic remains `smoothed_projection`.

```python
# Reuse near, far, support and device from the quick start above.
field, diagnostics = retrieval.run_restarts(
    (near, far), support=support, algorithm=retrieval.OSS,
    n_restarts=4, max_iter=2000, metric_every=20,
    algorithm_kwargs={'n_stages': 10, 'beta': 0.9},
    generator=torch.Generator(device=device).manual_seed(10),
)
print('OSS Fourier R factor:', diagnostics['r_factor_fourier'].cpu())
```

The default is a support-only **complex-field extension** suitable for speckles.
Measured near-field amplitude initializes the field but is not imposed in the
iterations. `enforce_magnitude=True` explicitly adds that two-plane constraint.
For the original real/nonnegative density model, set `positivity=True` instead;
this discards imaginary density components and uses HIO feedback for negative
projected density, including inside the support. These two options are exclusive.
The density mode requires only Fourier amplitudes and an object support: pass a
placeholder initialization amplitude such as `torch.ones_like(far)` as the first
magnitude, rather than supplying the unknown density. Raw OSS iterates retain
feedback outside the support; they are not support-clipped final projections.
Use `transforms.apply_support(field, support)` explicitly when extracting the
object-only field; its residual can differ from that of the raw iteration state.

The framework supplies:

- A default ten-stage schedule with Fourier-bin bandwidth `(alpha_y,alpha_x)`
  decreasing linearly from `(H,W)` to `(1/H,1/W)`. Rectangular grids use independent
  axis widths. The paper used square arrays, 2000 updates, and 100 random starts;
  counts and the rectangular generalization are configurable here.
- Equal stage lengths when possible; otherwise the earliest stages receive one
  extra update. Short runs use `min(n_stages,max_iter)` nonempty stages. A single
  stage uses the initial broad filter. `oss_bandwidth()` exposes this schedule.
- Best-candidate handoff at each stage boundary, independently per batch/channel.
  Selection uses `R_F = sum(abs(target - gamma*predicted))/sum(target)`, where
  `gamma=sum(target*predicted)/sum(predicted**2)` is our explicit least-squares
  scale convention. The paper specifies a scale factor but not its estimator.
  Only candidates produced within that stage compete; the last stage's best is
  returned by default. No reference field influences selection.
- `oss_stage_bounds` (zero-based, stop exclusive), `oss_stage_alpha`,
  `oss_stage_best_r_factor`, and `oss_stage_best_iteration` diagnostics in `run()`
  results or in `run_restarts()`'s `restart_results`. Best iteration numbers are
  one-based global update numbers. `r_factor_fourier` is also included in sampled
  metric histories. Zero-update runs return initialization and empty stage data.

`run_restarts()` ranks OSS's final-stage candidates by R_F, whereas other
algorithms retain their existing global-MSE selection. Its `mse_fourier` diagnostic
still reports actual MSE, even for OSS. In a direct `run()` call,
`return_best=True` explicitly overrides the last-stage result with the global MSE
winner; this is not the published OSS selection procedure.

For a single update or manual loop, `OSS(..., alpha=...)` accepts a scalar or
`(alpha_y,alpha_x)` in **Fourier bins**, not a spatial Gaussian sigma. Without a
fixed alpha, pass zero-based `iteration` and the total `max_iter`; manual loops
must manage stage handoffs themselves. `run()` and `run_restarts()` do this
management automatically, including with keyword-bound `functools.partial(OSS, ...)`.
An explicit object support is required. The removed heuristic's `sigma` and
`filter_decay` arguments belong to `smoothed_projection`, not actual OSS.

The Fourier Gaussian cache holds 16 filters and can be released with
`transforms.clear_fourier_gaussian_cache()`. R_F is defined as one for an all-zero
prediction against a nonzero target; for a zero target it returns unscaled L1
amplitude residual. Neither a low residual nor the original paper's comparisons
establishes improved recovery for every complex speckle dataset.

## Propagation and sampling

- Centered unitary transforms use `fftshift(fft2(ifftshift(x), norm='ortho'))`
  and the corresponding inverse, on the last two dimensions.
- `angular_spectrum_transform(field, dz=..., wavelength=..., pixel_size=...)`
  uses the scalar Helmholtz longitudinal wave number. Evanescent modes are
  discarded by default. `evanescent='decay'` models their forward attenuation;
  backward amplification is deliberately rejected in that mode.
- `fresnel_propagation(...)` uses the paraxial transfer function instead.
- Wavelength, distance, and pixel spacing are in meters. Pixel spacing may be
  a scalar or `(dy,dx)`; rectangular and odd-sized grids are supported.
- Propagation retains the sampled grid without automatic padding or resampling.
  Choose the field of view, spacing, and propagation distance to avoid periodic
  wraparound and aliasing. Backward propagation only inverts retained modes.
- Static transfer functions and Gaussian kernels have bounded caches (16 entries
  each). `clear_propagation_cache()` and `clear_gaussian_cache()` release them.
  Cached transfer tensors are read-only by convention. For differentiable physical
  parameters, build a transfer tensor explicitly and pass it as `transfer_function`.

The distinction between angular-spectrum and Fresnel propagation follows the
[scalar diffraction formulation](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html).

## Migration from earlier PyTorch notebooks

- `quality()` now detects amplitude-profile errors. Use
  `weighted_phase_agreement()` to reproduce the old definition.
- The optimizer defaults to normalized amplitude MSE instead of Pearson and has
  no default learning-rate scheduler or console logging. Set
  `loss_type='pearson_amplitude'` to retain the previous objective with corrected
  Pearson normalization. `log_every` enables periodic logs.
- An optional `scheduler_factory(optimizer)` can supply a learning-rate scheduler;
  it is stepped every `scheduler_every` optimizer updates (default 10). Plateau
  schedulers require scalar transfers at that interval; fixed schedules avoid them.
- `cyclic_loss=True` now actually enforces the measured far-field amplitude before
  inverse propagation, and returned fields always respect the support.
- `HIO` now uses nonzero feedback outside the support. Its default is support-only
  HIO; `enforce_magnitude=True` imposes near-field amplitude inside the support too.
- Odd-sized FFT centering has changed to the conventional centered-origin formula.
  Regenerate odd-sized synthetic Fourier data rather than mixing it with the old
  NumPy convention. Even-sized unpadded transforms retain the previous convention.
- `fresnel_transform` retains the historical angular-spectrum meaning with a
  deprecation warning; its legacy `propagator` argument remains a phase in radians.
  New code should use the explicitly named propagation functions above.
- `OSS` now implements oversampling smoothness with staged Fourier filtering.
  Use `smoothed_projection` to retain the earlier global-blur heuristic; see the
  OSS section for the distinction between density and complex-field constraints.
- `spectral`, `truncated_spectral`, and `orthogonal` are deprecated heuristic names.
  Use `autocorrelation_svd`, `thresholded_random_phase`, and
  `random_phase_multistart`, respectively. In the thresholded initializer,
  `drop_fraction=0.25` discards the weakest 25%; the old implementation retained
  only the strongest 25%. The unused legacy `n_iter` argument has no effect.
- Operator-based power iteration and fixed-step intensity-gradient methods remain
  experimental. Their default operators are now a centered unitary FFT/adjoint
  pair; their applicability depends on the actual measurement model.

## Validation and benchmarks

```bash
python -m pytest -q
python -m benchmarks.torch_reconstruction --method phase --device cpu
python -m benchmarks.torch_reconstruction --method raar --device cuda --restarts 4
python -m benchmarks.torch_reconstruction --method lbfgs --iterations 20 --lbfgs-inner-iterations 5
python -m benchmarks.torch_reconstruction --method hybrid --optimizer lbfgs --refinement-iterations 20
# Noisy two-plane counts, soft amplitude, and missing far-field pixels:
python -m benchmarks.torch_reconstruction --method hybrid --loss poisson --photons-per-peak 1000 --optimize-amplitude --detector-mask-fraction 0.1
# Original real/nonnegative OSS mode with Poisson-corrupted intensities:
python -m benchmarks.torch_reconstruction --method oss --object-type positive --iterations 2000 --photons-per-peak 1000
# Sensitivity to intensity noise and a one-pixel far-field calibration error:
python -m benchmarks.torch_reconstruction --method raar --noise-std 0.001 --shift-x 1
```

Tests cover numerical identities, analytic tilted-plane-wave propagation,
projection updates, cyclic gradients, support enforcement, rectangular/batched
fields, per-image selection, and seeded reconstruction. CUDA parity checks run
when CUDA is available. CPU CI covers Python 3.10 and 3.12.

The benchmark emits JSON with configuration, versions, synchronized wall time,
per-image raw measured and ideal Fourier NMSE, support-masked Fourier NMSE,
scaled Fourier R factor, and overlap with the known synthetic field inside the
support. Success fraction is explicitly based on the raw measurement NMSE threshold;
it is not a claim of unique or experimentally verified phase recovery. An
`observed_nmse` value also excludes masked detector pixels. When Poisson noise is
requested, both planes receive noisy counts with explicit per-plane gains. AD
reports include objective-evaluation counts and initial/final objectives; compare
elapsed time and objective quality rather than equating L-BFGS outer updates
with Adam updates. Hybrid timings include both projection and refinement.
Timing includes reconstruction diagnostics but excludes synthetic data generation.
Compare configurations on the same machine; the benchmark makes no GPU speedup
claim. Experimental calibration and real-data validation remain separate work.

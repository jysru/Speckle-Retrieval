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
- `OSS` is a deprecated alias for `smoothed_projection`, an experimental global
  smoothing heuristic, **not** the [published OSS algorithm](https://arxiv.org/abs/1211.4519).
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
# Sensitivity to intensity noise and a one-pixel far-field calibration error:
python -m benchmarks.torch_reconstruction --method raar --noise-std 0.001 --shift-x 1
```

Tests cover numerical identities, analytic tilted-plane-wave propagation,
projection updates, cyclic gradients, support enforcement, rectangular/batched
fields, per-image selection, and seeded reconstruction. CUDA parity checks run
when CUDA is available. CPU CI covers Python 3.10 and 3.12.

The benchmark emits JSON with configuration, versions, synchronized wall time,
per-image measured and ideal Fourier NMSE, and overlap with the known synthetic
field. Success fraction is explicitly based on the measurement NMSE threshold;
it is not a claim of unique or experimentally verified phase recovery. Timing
includes reconstruction diagnostics but excludes synthetic data generation.
Compare configurations on the same machine; the benchmark makes no GPU speedup
claim. Experimental calibration and real-data validation remain separate work.

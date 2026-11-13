"""Seeded CPU/CUDA reconstruction benchmark; run from the repository root.

python -m benchmarks.torch_reconstruction --method phase --device cpu
python -m benchmarks.torch_reconstruction --method raar --device cuda --restarts 4
python -m benchmarks.torch_reconstruction --method oss --object-type positive --iterations 2000

Success means meeting the measurement NMSE threshold, not unique phase recovery.
Timings include reconstruction and diagnostic metrics, exclude data generation,
and synchronize CUDA before and after timing. No cross-device speedup is claimed.
"""
import argparse
import json
import platform
import time

import torch

from speckleret.torch import initializers, metrics, retrieval, supports, transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--method', choices=['phase', 'raar', 'oss'], default='phase')
    parser.add_argument('--object-type', choices=['complex', 'positive'], default='complex')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--restarts', type=int, default=4, help='RAAR/OSS only')
    parser.add_argument('--oss-enforce-magnitude', action='store_true',
                        help='Opt into the two-plane complex OSS extension')
    parser.add_argument('--oss-stages', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1, help='CPU intra-op threads')
    parser.add_argument('--success-nmse', type=float, default=1e-3)
    parser.add_argument('--noise-std', type=float, default=0.,
                        help='Additive Gaussian intensity noise, relative to each peak; clip at zero')
    parser.add_argument('--photons-per-peak', type=float, default=0.,
                        help='Optional Poisson noise; expected count at the brightest pixel')
    parser.add_argument('--shift-y', type=int, default=0, help='Far-field calibration shift in pixels')
    parser.add_argument('--shift-x', type=int, default=0, help='Far-field calibration shift in pixels')
    args = parser.parse_args()
    if min(args.batch, args.size, args.seeds, args.iterations, args.restarts, args.threads, args.oss_stages) < 1:
        parser.error('Sizes, counts and iterations must be positive')
    if args.noise_std < 0 or args.photons_per_peak < 0:
        parser.error('Noise levels must be nonnegative')
    if args.object_type == 'positive' and args.oss_enforce_magnitude:
        parser.error('Positive-density OSS and measured-amplitude OSS are distinct modes')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')

    def reconstruct(near, far, mask, generator, iterations):
        if args.method == 'phase':
            return initializers.gradient_descent(
                near, far, mask, max_iter=iterations, initialization='random', generator=generator)
        algorithm = retrieval.OSS if args.method == 'oss' else retrieval.RAAR
        options = ({'positivity': args.object_type == 'positive',
                    'enforce_magnitude': args.oss_enforce_magnitude,
                    'n_stages': args.oss_stages} if args.method == 'oss' else {})
        out, _ = retrieval.run_restarts(
            (near, far), mask, n_restarts=args.restarts, generator=generator,
            algorithm=algorithm, algorithm_kwargs=options,
            max_iter=iterations, metric_every=20)
        return out

    samples, timings = [], []
    for seed in range(args.seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        if args.object_type == 'positive':
            target = torch.rand(args.batch, 1, args.size, args.size,
                                device=device, generator=generator).to(torch.complex64)
        else:
            target = torch.randn(args.batch, 1, args.size, args.size,
                                 dtype=torch.complex64, device=device, generator=generator)
        mask = supports.disk_support(target, radius=args.size / 4)
        target = transforms.apply_support(target, mask)
        near, ideal_far = target.abs(), transforms.fourier_transform(target).abs()
        if args.method == 'oss' and not args.oss_enforce_magnitude:
            # Support-only OSS does not assume measured object amplitudes.
            near = torch.ones_like(near)
        far = ideal_far
        if args.photons_per_peak:
            intensity = far.square()
            peak = intensity.amax((-2, -1), keepdim=True)
            counts = torch.poisson(intensity / peak * args.photons_per_peak, generator=generator)
            far = (counts * peak / args.photons_per_peak).sqrt()
        if args.noise_std:
            intensity = far.square()
            noise = torch.randn(intensity.shape, dtype=intensity.dtype, device=device, generator=generator)
            far = (intensity + args.noise_std * intensity.amax((-2, -1), keepdim=True) * noise).clamp_min(0).sqrt()
        far = torch.roll(far, (args.shift_y, args.shift_x), dims=(-2, -1))
        if seed == 0:
            reconstruct(near, far, mask, torch.Generator(device=device).manual_seed(999), 3)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = reconstruct(near, far, mask, torch.Generator(device=device).manual_seed(seed+1000), args.iterations)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        nmse = metrics.normalized_mse(transforms.fourier_transform(out), far).flatten().cpu().tolist()
        ideal_nmse = metrics.normalized_mse(transforms.fourier_transform(out), ideal_far).flatten().cpu().tolist()
        overlap = metrics.quality(out, target, mask=mask).flatten().cpu().tolist()
        r_factor = metrics.fourier_r_factor(transforms.fourier_transform(out), far).flatten().cpu().tolist()
        supported_nmse = metrics.normalized_mse(
            transforms.fourier_transform(transforms.apply_support(out, mask)), far).flatten().cpu().tolist()
        samples.extend(dict(seed=seed, sample=i, nmse=error, ideal_nmse=ideal_error,
                            supported_nmse=supported_error, r_factor_fourier=rf, field_overlap=q)
                       for i, (error, ideal_error, supported_error, rf, q) in enumerate(
                           zip(nmse, ideal_nmse, supported_nmse, r_factor, overlap)))

    report = dict(configuration=vars(args), python=platform.python_version(),
                  torch=torch.__version__, device_name=(torch.cuda.get_device_name(device)
                  if device.type == 'cuda' else platform.processor()),
                  seconds_per_batch=timings,
                  mean_seconds_per_sample=sum(timings)/(args.batch*args.seeds),
                  measurement_success_fraction=sum(s['nmse'] < args.success_nmse for s in samples)/len(samples),
                  samples=samples)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

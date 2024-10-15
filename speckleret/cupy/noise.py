import cupy as cp

intens_noise = 0*0.003
wavelength = 1064e-9
phase_noise = 0*wavelength/30


def add_intensity_noise(field, noise_std: float = 0):
    if noise_std == 0:
        return field
    else:
        intens = cp.square(cp.abs(field))
        max_intens = cp.max(intens)
        intens = max_intens * cp.abs(intens / max_intens + noise_std * cp.random.randn(*intens.shape))
        return cp.sqrt(intens) * cp.exp(1j * cp.angle(field))


def add_phase_noise(field, noise_std: float = 0, wavelength=1064e-9):
    if noise_std == 0:
        return field
    else:
        phase = cp.angle(cp.exp(1j * (cp.angle(field) + 2 * cp.pi / wavelength * noise_std * cp.random.randn(*field.shape))))
        return cp.abs(field) * cp.exp(1j * phase)


def add_complex_noise(field, intens_noise_std: float = 0, phase_noise_std: float = 0, wavelength: float = 1064e-9):
    if intens_noise_std != 0:
        field = add_intensity_noise(field, noise_std=intens_noise_std)
    if phase_noise_std != 0:
        field = add_phase_noise(field, noise_std=phase_noise_std, wavelength=wavelength)
    return field
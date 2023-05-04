import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def complex_to_hsv(z, rmin, rmax, hue_start=0):
    # get amplidude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=True) + hue_start
    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp - rmin) / (rmax - rmin)
    return hsv_to_rgb(np.dstack((h,s,v)))


def complex_imshow(field, figsize: tuple[int,int] = (15,5), remove_ticks: bool = False):
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    pl0 = axs[0].imshow(np.abs(field), cmap='gray')
    pl1 = axs[1].imshow(np.angle(field), cmap='hsv')
    pl2 = axs[2].imshow(complex_to_hsv(field, rmin=0, rmax=np.max(np.abs(field))))
    pls = [pl0, pl1, pl2]

    _ = axs[0].set_title("Amplitude")
    _ = axs[1].set_title("Phase")
    _ = axs[2].set_title("Complex field")

    if remove_ticks:
        _ = [axs[i].set_xticks([]) for i in range(len(axs))]
        _ = [axs[i].set_yticks([]) for i in range(len(axs))]
    
    return (fig, axs, pls)


def compare_complex_fields(field1, field2, figsize: tuple[int,int] = (15,5), remove_ticks: bool = False):
    ref1, ref2 = list(map(int, field1.shape)), list(map(int, field2.shape))

    fig, axs = plt.subplots(1, 6, figsize=figsize)
    pl0 = axs[0].imshow(np.abs(field1), cmap='gray')
    pl1 = axs[1].imshow(np.abs(field2), cmap='gray')
    pl2 = axs[2].imshow(np.angle(field1 * np.exp(-1j * np.angle(field1[ref1[0]//2, ref1[1]//2]))), cmap='hsv')
    pl3 = axs[3].imshow(np.angle(field2 * np.exp(-1j * np.angle(field2[ref2[0]//2, ref2[1]//2]))), cmap='hsv')
    pl4 = axs[4].imshow(complex_to_hsv(field1 * np.exp(-1j * np.angle(field1[ref1[0]//2, ref1[1]//2])), rmin=0, rmax=np.max(np.abs(field1))), cmap='gray')
    pl5 = axs[5].imshow(complex_to_hsv(field2 * np.exp(-1j * np.angle(field2[ref2[0]//2, ref2[1]//2])), rmin=0, rmax=np.max(np.abs(field2))), cmap='gray')
    pls = [pl0, pl1, pl2, pl3, pl4, pl5]

    _ = axs[0].set_title("Amplitude 1")
    _ = axs[1].set_title("Amplitude 2")
    _ = axs[2].set_title("Phase 1")
    _ = axs[3].set_title("Phase 2")
    _ = axs[4].set_title("Complex field 1")
    _ = axs[5].set_title("Complex field 2")

    if remove_ticks:
        _ = [axs[i].set_xticks([]) for i in range(len(axs))]
        _ = [axs[i].set_yticks([]) for i in range(len(axs))]

    return (fig, axs, pls)



if __name__ == "__main__":
    field1 = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
    field2 = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)

    compare_complex_fields(field1, field2, remove_ticks=True)
    plt.show()
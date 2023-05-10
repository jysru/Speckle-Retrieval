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


def rgb_imshow(rgb_list: list[np.ndarray, np.ndarray, np.ndarray] = [None, None, None], normalize_channels: bool = True):
    # Crop the list if it is longer than the number of available color channels
    if len(rgb_list) > 3:
        rgb_list = rgb_list[:3]

    # Append None values if the list is too short
    if len(rgb_list) < 3:
        for _ in range(3 - len(rgb_list)):
            rgb_list.append(None)

    # Check if the list is full of None
    if all(elem is None for elem in rgb_list):
        print(f"RGB tuple is empty...")
        return
    
    # Detect non None list indexes and normalize by their maximum value
    idx_full = [i for i in range(len(rgb_list)) if rgb_list[i] is not None]
    if normalize_channels:
        for i in idx_full:
            rgb_list[i] = rgb_list[i] / np.max(rgb_list[i])

    # Assign zeros to None indexes
    idx_none = [i for i in range(len(rgb_list)) if rgb_list[i] is None]
    for i in idx_none:
        rgb_list[i] = np.zeros(rgb_list[idx_full[0]].shape)

    # RGB imshow from the three first elements in the rgb_list
    fig = plt.figure()
    plt.imshow(np.dstack(rgb_list))
    return fig


def phase_difference(field1: np.ndarray, field2: np.ndarray, support: np.ndarray = None, cmap: str = 'gray'):
    if support is not None:
        field1[np.logical_not(support)] = np.nan
        field2[np.logical_not(support)] = np.nan
    phi_diff = np.angle(np.exp(1j * (np.angle(field1) - np.angle(field2))))

    print(f"Phase error:")
    print(f"  - Mean: {np.nanmean(phi_diff):3.5f} rad")
    print(f"  - Std: {np.nanstd(phi_diff):3.5f} rad")

    plt.figure()
    plt.imshow(phi_diff - np.nanmean(phi_diff), cmap=cmap)
    plt.colorbar()
    plt.title("Phase difference")


def compare_arrays(array1, array2, figsize: tuple[int,int] = (15,5), remove_ticks: bool = False, remove_colorbars: bool = False, intensity: bool = False, cmap: str = 'gray'):
    if np.iscomplexobj(array1):
        array1 = np.abs(array1)
    if np.iscomplexobj(array2):
        array2 = np.abs(array2)
    if intensity:
        title_base_str = "Intensity"
        array1 = np.square(array1)
        array2 = np.square(array2)
    else:
        title_base_str = "Amplitude"

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    pl0 = axs[0].imshow(array1, cmap=cmap)
    pl1 = axs[1].imshow(array2, cmap=cmap)
    pls = [pl0, pl1]

    _ = axs[0].set_title(title_base_str + " 1")
    _ = axs[1].set_title(title_base_str + " 2")

    if not remove_colorbars:
        plt.colorbar(pl0, ax=axs[0])
        plt.colorbar(pl1, ax=axs[1])

    if remove_ticks:
        _ = [axs[i].set_xticks([]) for i in range(len(axs))]
        _ = [axs[i].set_yticks([]) for i in range(len(axs))]

    return (fig, axs, pls)


def compare_complex_fields(field1, field2, figsize: tuple[int,int] = (15,5), remove_ticks: bool = False, ref: tuple[int, int] = None):
    if ref is None:
        ref = np.argmax(np.abs(field1))
        ref = np.unravel_index(ref, field1.shape)

    fig, axs = plt.subplots(1, 6, figsize=figsize)
    pl0 = axs[0].imshow(np.abs(field1), cmap='gray')
    pl1 = axs[1].imshow(np.abs(field2), cmap='gray')
    pl2 = axs[2].imshow(np.angle(field1 * np.exp(-1j * np.angle(field1[ref[0], ref[1]]))), cmap='hsv')
    pl3 = axs[3].imshow(np.angle(field2 * np.exp(-1j * np.angle(field2[ref[0], ref[1]]))), cmap='hsv')
    pl4 = axs[4].imshow(complex_to_hsv(field1 * np.exp(-1j * np.angle(field1[ref[0], ref[1]])), rmin=0, rmax=np.max(np.abs(field1))), cmap='gray')
    pl5 = axs[5].imshow(complex_to_hsv(field2 * np.exp(-1j * np.angle(field2[ref[0], ref[1]])), rmin=0, rmax=np.max(np.abs(field2))), cmap='gray')
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


def compare_complex_planes(field1, field2, field3, field4, figsize: tuple[int,int] = (15,8), planes_names: tuple[str, str] = ('Plane 1', 'Plane 2'), remove_ticks: bool = False):
    ref1, ref2, ref3, ref4 = list(map(int, field1.shape)), list(map(int, field2.shape)), list(map(int, field3.shape)), list(map(int, field4.shape))

    fig, axs = plt.subplots(2, 6, figsize=figsize)
    pl0 = axs[0,0].imshow(np.abs(field1), cmap='gray')
    pl1 = axs[0,1].imshow(np.abs(field2), cmap='gray')
    pl2 = axs[0,2].imshow(np.angle(field1 * np.exp(-1j * np.angle(field1[ref1[0]//2, ref1[1]//2]))), cmap='hsv')
    pl3 = axs[0,3].imshow(np.angle(field2 * np.exp(-1j * np.angle(field2[ref2[0]//2, ref2[1]//2]))), cmap='hsv')
    pl4 = axs[0,4].imshow(complex_to_hsv(field1 * np.exp(-1j * np.angle(field1[ref1[0]//2, ref1[1]//2])), rmin=0, rmax=np.max(np.abs(field1))), cmap='gray')
    pl5 = axs[0,5].imshow(complex_to_hsv(field2 * np.exp(-1j * np.angle(field2[ref2[0]//2, ref2[1]//2])), rmin=0, rmax=np.max(np.abs(field2))), cmap='gray')

    pl6 = axs[0,0].imshow(np.abs(field3), cmap='gray')
    pl7 = axs[0,1].imshow(np.abs(field4), cmap='gray')
    pl8 = axs[0,2].imshow(np.angle(field3 * np.exp(-1j * np.angle(field3[ref3[0]//2, ref3[1]//2]))), cmap='hsv')
    pl9 = axs[0,3].imshow(np.angle(field4 * np.exp(-1j * np.angle(field4[ref4[0]//2, ref4[1]//2]))), cmap='hsv')
    pl10 = axs[0,4].imshow(complex_to_hsv(field3 * np.exp(-1j * np.angle(field3[ref3[0]//2, ref3[1]//2])), rmin=0, rmax=np.max(np.abs(field3))), cmap='gray')
    pl11 = axs[0,5].imshow(complex_to_hsv(field4 * np.exp(-1j * np.angle(field4[ref4[0]//2, ref4[1]//2])), rmin=0, rmax=np.max(np.abs(field4))), cmap='gray')

    pls = [pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8, pl9, pl10, pl11]

    for j in range(len(axs)):
        _ = axs[j,0].set_title("Amplitude 1" + "\n" + planes_names[j])
        _ = axs[j,1].set_title("Amplitude 2" + "\n" + planes_names[j])
        _ = axs[j,2].set_title("Phase 1" + "\n" + planes_names[j])
        _ = axs[j,3].set_title("Phase 2" + "\n" + planes_names[j])
        _ = axs[j,4].set_title("Complex field 1" + "\n" + planes_names[j])
        _ = axs[j,5].set_title("Complex field 2" + "\n" + planes_names[j])

    if remove_ticks:
        _ = [axs[j,i].set_xticks([]) for i in range(len(axs[0])) for j in range(len(axs))]
        _ = [axs[j,i].set_yticks([]) for i in range(len(axs[0])) for j in range(len(axs))]

    return (fig, axs, pls)


if __name__ == "__main__":
    field1 = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
    field2 = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)

    compare_complex_fields(field1, field2, remove_ticks=True)
    plt.show()
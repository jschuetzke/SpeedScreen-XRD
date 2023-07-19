from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from powdiffrac.processing import scale_min_max
from powdiffrac.simulation import peak_shapes
from scipy.ndimage import binary_dilation, convolve1d

from background import chebyshev, decay


def generate_variations(
    x: ArrayLike,
    step_size: float = 0.015,
    background_type: str = "chebyshev",
    multi_peak_range: Union[Tuple[int, int], None] = None,
    max_multi_peak: float = 0.3,
    detection_threshold: float = 0.05,
    num_multi_peaks: int = 2,
    restricted_area: int = 25,
    fwhm_range: Tuple[float, float] = (0.2, 0.7),
    noise_lvl: Tuple[float, float] = (0.03, 0.07),
    sample_holder: Union[int, None] = None,
    random_shift: int = 0,
    seed: int = None,
):
    # patterns should be scaled between 0 and 1
    if np.max(x) > 1.0:
        x = scale_min_max(x)

    num_scans, datapoints = x.shape
    y = np.ones([num_scans])
    rng = np.random.default_rng(seed)

    # %% artifacts and augmentation that are not related to the respective classes

    # add random shift to scans
    if random_shift:
        shift_amount = rng.integers(-random_shift, random_shift)
        x_pad = np.pad(
            x, ((0, 0), (abs(shift_amount), abs(shift_amount))), constant_values=(0, 0)
        )
        start = shift_amount + abs(shift_amount)
        end = datapoints + start
        x = x_pad[:, start:end]

    # if sample holder is present, add peak
    if (type(sample_holder) is int) and (sample_holder > 0):
        # batch_x[:, self.rng.integers(self.holder-10, self.holder+10)] = self.rng.uniform(0.1, 0.15)
        holder = np.zeros_like(x)
        holder[:, rng.integers(sample_holder - 20, sample_holder + 20)] = 1.0
        kernel = peak_shapes.get_gaussian_kernel(0.3, step_size)
        holder = convolve1d(holder, kernel, mode="constant")
        holder /= np.max(holder, axis=1, keepdims=True)

    # generate typical baseline intensity
    if background_type == "decay":
        bkg = decay(x.shape)
    elif background_type == "chebyshev":
        bkg = chebyshev(x.shape)
        bkg *= 0.2
    else:
        raise ValueError(f"background type {background_type} unknown.")

    # %% artifacts and augmentation that are related to non-diffracting class

    # Non-diffracting -> No or minor peaks -> damping single-phase peaks
    damping_factor = rng.uniform(0.0, 0.2, size=num_scans)
    # decide whether sample contains structure or not
    labels = rng.choice([False, True], size=num_scans, p=[0.33, 0.67])
    # no damping for those samples with meaningful pattern
    damping_factor[labels] += 0.7
    # set labels for noise samples to 0
    y[np.invert(labels)] = 0.0

    # we sum bkg and scans (with damping) later, once peak shapes are convolved

    # %% Multi-phase augmentation

    # before we add the multiphase peaks, we have to draw random noise levels
    # ensure that noise does not cover multi-phase peaks
    noise_lvls = rng.uniform(noise_lvl[0], noise_lvl[1], size=(num_scans))

    # select indices of scans to transform into multiphase
    multi_indices = rng.choice(
        np.arange(num_scans),
        np.ceil(num_scans / 2).astype(int),
        replace=False,
    )
    # get position of single phase peaks in batch
    cur_peak_pos = np.sum(x, axis=0).astype(bool)
    # dilate positions to avoid too much overlap between single and multiphase peaks
    restricted = binary_dilation(cur_peak_pos, iterations=restricted_area)
    if multi_peak_range:
        restricted[: multi_peak_range[0]] = True
        restricted[multi_peak_range[1] :] = True
    else:
        restricted[:200] = True
        restricted[-150:] = True
    # get eligible positions for multiphase peaks
    elig_pos = np.arange(datapoints)[np.invert(restricted)]

    # position of multiphase peaks
    add_pos = rng.choice(elig_pos, (multi_indices.size, num_multi_peaks), replace=True)
    peak_heights = np.random.uniform(
        detection_threshold, max_multi_peak, size=add_pos.shape
    )
    # slice and reshape noise level array to compare with multi-phase peak heights
    # ensure that extra peaks are detectable
    noise_comparison = np.repeat(
        noise_lvls[multi_indices, None], num_multi_peaks, axis=1
    )
    # identify multi-phase peaks too small for recognition (lost in noise)
    mask = np.any(peak_heights < noise_comparison * 2.0 / 3, axis=1)
    # correct height to ensure detectability
    peak_heights[mask] = noise_comparison[mask]

    # array for those multi-phase peaks
    add_peaks = np.zeros_like(x)

    # fill array
    add_peaks[multi_indices[:, None], add_pos] = peak_heights

    # combine additional peaks with signle-phase patterns

    x += add_peaks
    y[multi_indices] *= 2.0

    # %% convolve peak shapes

    fwhm = rng.uniform(fwhm_range[0], fwhm_range[1], size=num_scans)
    eta = rng.uniform(0.1, 0.9, size=num_scans)
    for n in np.arange(num_scans):
        kernel = peak_shapes.get_pseudo_voigt_kernel(fwhm[n], step_size, eta[n])
        x[n] = convolve1d(x[n], kernel, mode="constant")

    x = scale_min_max(x)

    # correct absolute intensities due to peak broadening
    x *= np.sqrt(np.maximum(1 - fwhm, 0.3))[:, None]
    # x *= np.maximum(1 - fwhm, 0.35)[:, None]

    # combine pattern and background
    x = x * damping_factor[:, None] + bkg

    # add sample holder peak if necessary
    if (type(sample_holder) is int) and (sample_holder > 0):
        x += holder * rng.uniform(0.03, 0.05)

    # add noise, clip extreme values
    gaus = 1 / 3 * np.clip(rng.normal(0, 1, x.shape), -3, 3)
    pois = 1 / 3 * np.clip(rng.normal(0, 1, x.shape), -3, 3)
    # next we shift the noise from -1 and 1 to 0 and 1
    gaus = (gaus * 0.5) + 0.5
    pois = (pois * 0.5) + 0.5
    # scale noise
    gaus *= noise_lvls[:, None]
    pois *= np.sqrt(x) * noise_lvls[:, None]

    x += gaus
    x += pois

    # only scale patterns with structure
    x[np.where(y >= 1.0)[0]] = scale_min_max(x[np.where(y >= 1.0)[0]])
    return x, y

#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>
##
## microAO is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## microAO is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with microAO.  If not, see <http://www.gnu.org/licenses/>.

#Import required packs
import dataclasses
import collections.abc
import numpy as np
from scipy.signal import tukey
from skimage.filters import threshold_otsu
import skimage.transform

import microAO.gui.metricDiagnostics


@dataclasses.dataclass(frozen=True)
class DiagnosticsFourier:
    fft_sq_log: np.ndarray
    freq_above_noise: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsContrast:
    image_raw: np.ndarray
    mean_top: float
    mean_bottom: float

@dataclasses.dataclass(frozen=True)
class DiagnosticsGradient:
    image_raw: np.ndarray
    grad_mask_x: np.ndarray
    grad_mask_y: np.ndarray
    correction_grad: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsFourierPower:
    fftarray_sq_log: np.ndarray
    freq_above_noise: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsSecondMoment:
    fftarray_sq_log: np.ndarray
    fftarray_sq_log_masked: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsFourierSI:
    fft_sq_log: np.ndarray
    SI_mask: np.ndarray
    freq_above_noise: np.ndarray


def mask_circular(dims, radius_inner=0, radius_outer=None, centre=None):
    # Ensure the dimensions and the centre point are numpy arrays
    dims = np.array(dims)
    centre = np.array(centre)
    # Initialise centre and outer radius if necessary
    if np.any(centre == None):
        centre = np.flip(dims) / 2
    if radius_outer is None:
        # Largest circle that could fit in the dimensions
        radius_outer = min(centre, dims - centre)
    # Create a meshgrid
    meshgrid = np.meshgrid(np.arange(dims[1]), np.arange(dims[0]))
    # Calculate distances from the centre element
    distance = np.sqrt(((meshgrid - centre.reshape(-1, 1, 1)) ** 2).sum(axis=0))
    # Return a binary mask for the specified radius
    return np.logical_and(radius_inner <= distance, distance <= radius_outer)

def measure_fourier_metric(image, wavelength, NA, pixel_size, noise_amp_factor=1.125, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.max(image.shape) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = mask_circular(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * noise_amp_factor

    OTF_mask = mask_circular(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask
    metric = np.count_nonzero(freq_above_noise)
    return metric, DiagnosticsFourier(fftarray_sq_log, freq_above_noise)

def measure_contrast_metric(image, no_intensities = 100, **kwargs):
    flattened_image = image.flatten()

    flattened_image_list = flattened_image.tolist()
    flattened_image_list.sort()

    mean_top = np.mean(flattened_image_list[-no_intensities:])
    mean_bottom = np.mean(flattened_image[:no_intensities])
    return (
        mean_top/mean_bottom,
        DiagnosticsContrast(
            image,
            mean_top,
            mean_bottom
        )
    )

def measure_gradient_metric(image, **kwargs):
    image_gradient_x = np.gradient(image, axis=1)
    image_gradient_y = np.gradient(image, axis=0)

    grad_mask_x = image_gradient_x > (threshold_otsu(image_gradient_x) * 1.125)
    grad_mask_y = image_gradient_y > (threshold_otsu(image_gradient_y) * 1.125)

    correction_grad = np.sqrt((image_gradient_x * grad_mask_x) ** 2 + (image_gradient_y * grad_mask_y) ** 2)

    metric = np.mean(correction_grad)
    return metric, DiagnosticsGradient(image, grad_mask_x, grad_mask_y, correction_grad)


def measure_fourier_power_metric(image, wavelength, NA, pixel_size, noise_amp_factor=1.125,
                                 high_f_amp_factor=100, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = freq_ratio * (np.max(np.shape(image)) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = mask_circular(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * noise_amp_factor

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[0]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[1]))
    ramp_mask = x_prime ** 2 + y_prime ** 2

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    dist = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                   np.arange(-rad_x, rad_x) ** 2)
    omega = 1 - np.exp((dist / OTF_outer_rad) - 1)

    high_f_amp_mask = high_f_amp_factor * (ramp_mask * omega) / np.max(ramp_mask * omega)

    OTF_mask = mask_circular(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask * high_f_amp_mask
    metric = np.sum(freq_above_noise)
    return metric, DiagnosticsFourierPower(fftarray_sq_log, freq_above_noise)


def measure_second_moment_metric(image, wavelength, NA, pixel_size, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = freq_ratio * (np.max(np.shape(image)) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    ring_mask = mask_circular(np.shape(image), 0, OTF_outer_rad)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[0]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[1]))
    ramp_mask = x_prime ** 2 + y_prime ** 2

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    dist = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                   np.arange(-rad_x, rad_x) ** 2)
    omega = 1 - np.exp((dist/OTF_outer_rad)-1)

    fftarray_sq_log_masked = ring_mask * fftarray_sq_log * ramp_mask * omega
    metric = np.sum(fftarray_sq_log_masked)/np.sum(fftarray_sq_log)
    return metric, DiagnosticsSecondMoment(fftarray_sq_log, fftarray_sq_log_masked)

def measure_fourier_SI_metric(image, wavelength, NA, pixel_size, noise_amp_factor=1.125, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.max(image.shape) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = mask_circular(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * noise_amp_factor

    OTF_mask = mask_circular(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)

    # The structured illumination mask is defined on the unit circle, i.e. a
    # square image. The radius is in the range [0.0, 1.0], where 1.0 is half
    # the image's side. The angle is in degrees, starting from the right-side
    # horizontal axis and increasing clockwise. Only the first (bottom) half
    # of the mask needs to be defined, as it is later mirrored along the
    # horizontal axis.
    SI_mask = np.zeros((max(image.shape), max(image.shape))).astype(bool)
    for r, theta in [(0.414 * r_factor, theta) for r_factor in (1, 2) for theta in (43.0, 102.5, 162.0)]:
        # Scale the radius to the image's dimensions
        r = r * max(SI_mask.shape) / 2
        x = r * np.cos(np.deg2rad(theta)) + SI_mask.shape[1] / 2
        y = r * np.sin(np.deg2rad(theta)) + SI_mask.shape[0] / 2
        SI_mask = np.logical_or(
            SI_mask,
            mask_circular(
                np.array(SI_mask.shape),
                0,
                0.02 * SI_mask.shape[0],
                (x, y)
            )
        )
    SI_mask = np.logical_or(SI_mask, np.flip(SI_mask))
    if image.shape[0] != image.shape[1]:
        # Interpolate
        SI_mask = skimage.transform.resize(
            SI_mask,
            image.shape
        )

    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask * SI_mask
    metric = np.count_nonzero(freq_above_noise)
    return metric, DiagnosticsFourierSI(fftarray_sq_log, SI_mask, freq_above_noise)

@dataclasses.dataclass(frozen=True)
class _MetricObjects:
    function: collections.abc.Callable
    diagnostic_panel: microAO.gui.metricDiagnostics.DiagnosticsPanelBase

metrics = {
    "Fourier": _MetricObjects(
        measure_fourier_metric,
        microAO.gui.metricDiagnostics.DiagnosticsPanelFourier
    ),
    "Contrast": _MetricObjects(
        measure_contrast_metric,
        microAO.gui.metricDiagnostics.DiagnosticsPanelContrast
    ),
    "Fourier power": _MetricObjects(
        measure_fourier_power_metric,
        microAO.gui.metricDiagnostics.DiagnosticsPanelFourierPower
    ),
    "Gradient": _MetricObjects(
        measure_gradient_metric,
        microAO.gui.metricDiagnostics.DiagnosticsPanelGradient
    ),
    "Second moment": _MetricObjects(
        measure_second_moment_metric,
        microAO.gui.metricDiagnostics.DiagnosticsPanelSecondMoment
    ),
    "Fourier SI": _MetricObjects(
        measure_fourier_SI_metric,
        microAO.gui.metricDiagnostics.DiagnosticsPanelFourierSI
    ),
}

"""Gaussian model fitting for the camera-derived, normalized 1D spot profiles.

This module does not modify the existing 1/e^2 width finder. Camera response
is not radiometrically calibrated, so fitted radii are *apparent* radii in px.
"""

import numpy as np
from scipy.optimize import curve_fit


def gaussian_profile(axis, amplitude, center, radius, offset):
    """I(x) = A * exp(-2 * ((x-x0)/w)^2) + B."""
    x = np.asarray(axis, dtype=float)
    return amplitude * np.exp(-2.0 * ((x - center) / radius) ** 2) + offset


def fit_gaussian_profile(axis, profile, radius_guess=None):
    """Fit one normalized profile with a free constant background offset.

    Parameters
    ----------
    axis, profile : 1D arrays
        Pixel coordinates and corresponding (possibly background-corrected)
        camera signal. They must have equal lengths and finite values.
    radius_guess : float, optional
        Initial 1/e^2 radius in pixel units, e.g. from Beam_W_Finder.

    Returns
    -------
    dict
        radius, center, amplitude, offset, fitted_profile, r_squared, rmse,
        relative_width_difference (when a radius_guess was supplied).

    Raises
    ------
    ValueError, RuntimeError
        When the profile is unsuitable or the least-squares fit fails.
    """
    x = np.asarray(axis, dtype=np.float64)
    y = np.asarray(profile, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 9:
        raise ValueError("Need at least 9 matching 1D profile samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Profile contains non-finite values")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Axis coordinates must be strictly increasing")

    span = float(x[-1] - x[0])
    step = float(np.median(np.diff(x)))
    edge_count = max(3, min(x.size // 10, 12))
    edge_samples = np.concatenate((y[:edge_count], y[-edge_count:]))
    offset_guess = float(np.median(edge_samples))
    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    amplitude_guess = peak - offset_guess
    noise_proxy = float(np.std(edge_samples))

    if amplitude_guess <= max(0.025, 3.0 * noise_proxy):
        raise ValueError("No sufficiently prominent peak for Gaussian fit")
    # Both tails must be observable. The fit is not trustworthy when the beam
    # continues right up to the crop boundary.
    if peak_idx <= 1 or peak_idx >= x.size - 2:
        raise ValueError("Peak lies too close to the profile boundary")

    if radius_guess is not None and np.isfinite(radius_guess) and radius_guess > 0:
        starting_radius = float(radius_guess)
    else:
        baseline_corrected = np.maximum(y - offset_guess, 0.0)
        weight_sum = float(baseline_corrected.sum())
        variance = (float(np.sum(baseline_corrected * (x - x[peak_idx]) ** 2))
                    / weight_sum) if weight_sum > 0 else (span / 8) ** 2
        starting_radius = 2.0 * np.sqrt(max(variance, step * step))

    min_radius = max(step * 0.5, 0.25)
    max_radius = max(span * 2.0, 10 * min_radius)
    starting_radius = float(np.clip(starting_radius, min_radius * 1.01, max_radius * 0.99))
    scale = max(float(np.ptp(y)), 0.1)
    lower = [0.0, float(x[0]), min_radius, float(y.min() - scale)]
    upper = [4.0 * scale + amplitude_guess, float(x[-1]), max_radius,
             float(y.max() + scale)]

    try:
        params, _ = curve_fit(
            gaussian_profile, x, y,
            p0=[amplitude_guess, float(x[peak_idx]), starting_radius, offset_guess],
            bounds=(lower, upper),
            maxfev=3000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ValueError(f"Gaussian fit failed: {exc}") from exc

    amplitude, center, radius, offset = map(float, params)
    fit = gaussian_profile(x, *params)
    residuals = y - fit
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    if ss_tot <= 1e-12:
        raise ValueError("Profile has no measurable variation")

    relative_diff = None
    if radius_guess is not None and np.isfinite(radius_guess) and radius_guess > 0:
        relative_diff = (radius - float(radius_guess)) / float(radius_guess)

    return {
        "radius": radius,
        "center": center,
        "amplitude": amplitude,
        "offset": offset,
        "fitted_profile": fit,
        "r_squared": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "relative_width_difference": relative_diff,
    }


def fit_width_result(width_result):
    """Fit both X/Y profiles returned by measure_beam_width."""
    if width_result is None:
        raise ValueError("No width measurement supplied")
    result = {}
    for axis_key, radius_key in (("x_profile", "wx"), ("y_profile", "wy")):
        info = width_result[axis_key]
        result[axis_key[0]] = fit_gaussian_profile(
            info["axis"], info["profile"],
            radius_guess=float(width_result[radius_key]),
        )
    return result

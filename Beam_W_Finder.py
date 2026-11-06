import numpy as np
from scipy.ndimage import gaussian_filter1d


def find_1e2_radius(axis, profile, center_guess, background):

    # Step 1: Subtract supplied background
    corrected = np.asarray(profile, dtype=float) - background

    # Estimate residual baseline from profile edges
    edge_size = max(3, min(10, len(corrected) // 5))

    edge_values = np.concatenate((
        corrected[:edge_size],
        corrected[-edge_size:]
    ))

    residual_baseline = np.median(edge_values)

    # Remove residual noise floor
    corrected = corrected - residual_baseline

    # Step 2: Smooth the corrected 1D profile
    smooth = gaussian_filter1d(corrected, sigma=1)

    # Step 3: Locate the maximum near the beam center
    center_idx = np.argmin(np.abs(axis - center_guess))

    search_start = max(0, center_idx - 10)
    search_end = min(len(axis), center_idx + 11)

    peak_idx = search_start + np.argmax(
        smooth[search_start:search_end]
    )

    peak = smooth[peak_idx]

    if peak <= 0:
        raise ValueError("No positive beam signal")

    # Step 4: Find the 1/e^2 intensity level
    threshold = peak * np.exp(-2)

    # Step 5: Find left crossing
    left = peak_idx

    while left > 0 and smooth[left] > threshold:
        left -= 1

    if left == 0 and smooth[left] > threshold:
        raise ValueError("Beam truncated on left side")

    # Interpolate between neighboring samples
    x1, x2 = axis[left], axis[left + 1]
    i1, i2 = smooth[left], smooth[left + 1]

    left_position = x1 + (
        (threshold - i1) * (x2 - x1) / (i2 - i1)
    )

    # Step 6: Find right crossing
    right = peak_idx

    while right < len(axis) - 1 and smooth[right] > threshold:
        right += 1

    if right == len(axis) - 1 and smooth[right] > threshold:
        raise ValueError("Beam truncated on right side")

    x1, x2 = axis[right - 1], axis[right]
    i1, i2 = smooth[right - 1], smooth[right]

    right_position = x1 + (
        (threshold - i1) * (x2 - x1) / (i2 - i1)
    )

    # Step 7: Calculate beam radius
    radius = (right_position - left_position) / 2

    return {
        "radius": radius,
        "left": left_position,
        "right": right_position,
        "axis": axis,
        "profile": smooth / peak
    }


def measure_beam_width(I, x, y, xc, yc, background):

    # Average over the Y axis to get X profile
    profile_x = np.mean(I, axis=0)

    # Average over the X axis to get Y profile
    profile_y = np.mean(I, axis=1)

    result_x = find_1e2_radius(
        x, profile_x, xc, background
    )

    result_y = find_1e2_radius(
        y, profile_y, yc, background
    )

    return {
        "wx": result_x["radius"],
        "wy": result_y["radius"],
        "x_profile": result_x,
        "y_profile": result_y
    }

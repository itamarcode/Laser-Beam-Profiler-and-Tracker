import numpy as np
from scipy.ndimage import gaussian_filter, label

def estimate_background(image, edge_width=30):

    # Collect pixels from the image borders
    edges = np.concatenate([
        image[:edge_width, :].ravel(),
        image[-edge_width:, :].ravel(),
        image[:, :edge_width].ravel(),
        image[:, -edge_width:].ravel()
    ])

    # Estimate constant background using median
    background = np.median(edges)

    # Estimate random noise using MAD
    mad = np.median(np.abs(edges - background))
    noise_std = 1.4826 * mad

    # Fallback for strongly clipped distributions
    if noise_std < 1e-10:
        noise_std = np.std(edges)

    return background, noise_std

def refine_center(
    I_full, x, y, roi_bounds,
    background, noise_std,
    max_iter=10, tolerance=0.001
):

    X, Y = np.meshgrid(x, y)

    # ------------------------------------
    # 1. Background correction
    # ------------------------------------

    I_corrected = I_full - background

    # Smooth only for signal detection
    I_smooth = gaussian_filter(
        I_corrected, sigma=1.5
    )

    # ------------------------------------
    # 2. Initial position from Stage 1 ROI
    # ------------------------------------

    r0, r1, c0, c1 = roi_bounds

    local_smooth = I_smooth[r0:r1, c0:c1]

    peak_r, peak_c = np.unravel_index(
        np.argmax(local_smooth),
        local_smooth.shape
    )

    peak_r += r0
    peak_c += c0

    xc = x[peak_c]
    yc = y[peak_r]

    peak = I_smooth[peak_r, peak_c]

    if peak <= 5 * noise_std:
        raise ValueError("Beam not detected above noise")

    # ------------------------------------
    # 3. Create signal mask
    # ------------------------------------

    threshold = max(
        0.02 * peak,
        2.0 * noise_std
    )

    signal_mask = I_smooth > threshold

    # Keep the connected region containing
    # the detected beam peak
    regions, count = label(signal_mask)

    region_id = regions[peak_r, peak_c]

    if region_id == 0:
        raise ValueError("No valid beam region")

    signal_mask = regions == region_id

    # Use original, unsmoothed intensity
    weights = np.where(
        signal_mask,
        np.maximum(I_corrected, 0),
        0.0
    )

    # ------------------------------------
    # 4. Iterative center refinement
    # ------------------------------------

    roi_width_x = x[c1 - 1] - x[c0]
    roi_width_y = y[r1 - 1] - y[r0]

    half_width_x = 1.5 * roi_width_x
    half_width_y = 1.5 * roi_width_y

    center_history = [(xc, yc)]

    for k in range(max_iter):

        window = (
            (np.abs(X - xc) <= half_width_x) &
            (np.abs(Y - yc) <= half_width_y)
        )

        W = np.where(window, weights, 0)

        total = np.sum(W)

        if total <= 0:
            raise ValueError("No signal inside window")

        xc_new = np.sum(X * W) / total
        yc_new = np.sum(Y * W) / total

        shift = np.hypot(
            xc_new - xc,
            yc_new - yc
        )

        center_history.append((xc_new, yc_new))

        print(
            f"Iteration {k+1}: "
            f"X={xc_new:.5f}, "
            f"Y={yc_new:.5f}, "
            f"Shift={shift:.6f} mm"
        )

        xc, yc = xc_new, yc_new

        if shift < tolerance:
            break

    return {
        "center_x": xc,
        "center_y": yc,
        "center_history": np.array(center_history),
        "background": background,
        "noise_std": noise_std,
        "I_corr": I_corrected
    }
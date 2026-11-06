import numpy as np

def generate_beam(I0=1.0, w=1.0, x0=1.0, y0=-3.0,
                  background_level=0.03, noise_std=0.02,
                  x_min=-9, x_max=9, y_min=-9, y_max=9, n_points=601, seed=None):

    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)

    X, Y = np.meshgrid(x, y)

    I_ideal = I0 * np.exp(
        -2 * (((X - x0)**2 + (Y - y0)**2) / w**2)
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, I_ideal.shape)

    I_raw = I_ideal + background_level + noise
    I_raw = np.clip(I_raw, 0, None)

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "I_ideal": I_ideal,
        "I_raw": I_raw,
        "params": {
            "I0": I0,
            "w": w,
            "x0": x0,
            "y0": y0,
            "background_level": background_level,
            "noise_std": noise_std
        }
    }


if __name__ == "__main__":
    result = generate_beam()
    print("Beam generated successfully.")
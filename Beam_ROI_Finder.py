
import numpy as np


def find_recursive_roi(
    I, x, y, iterations=3,
    grid_sizes=None, anchor=None
):

    if grid_sizes is None:
        grid_sizes = [4] * iterations

    if len(grid_sizes) != iterations:
        raise ValueError(
            "grid_sizes must match iterations"
        )

    r0, r1 = 0, I.shape[0]
    c0, c1 = 0, I.shape[1]

    if anchor is not None:
        anchor_row, anchor_col = anchor

        if not (
            0 <= anchor_row < I.shape[0]
            and 0 <= anchor_col < I.shape[1]
        ):
            raise ValueError("Anchor outside image")

    history = []

    for n in range(iterations):

        grid_size = grid_sizes[n]

        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")

        if (
            r1 - r0 < grid_size
            or c1 - c0 < grid_size
        ):
            raise ValueError("ROI too small")

        row_edges = np.linspace(
            r0, r1, grid_size + 1, dtype=int
        )

        col_edges = np.linspace(
            c0, c1, grid_size + 1, dtype=int
        )

        means = np.zeros(
            (grid_size, grid_size)
        )

        for i in range(grid_size):
            for j in range(grid_size):

                cell = I[
                    row_edges[i]:row_edges[i + 1],
                    col_edges[j]:col_edges[j + 1]
                ]

                means[i, j] = np.mean(cell)

        scores = (
            means[:-1, :-1]
            + means[1:, :-1]
            + means[:-1, 1:]
            + means[1:, 1:]
        )

        # Restrict candidate windows to those
        # containing the detected global peak
        if anchor is not None:

            valid_rows = (
                (row_edges[:-2] <= anchor_row)
                & (anchor_row < row_edges[2:])
            )

            valid_cols = (
                (col_edges[:-2] <= anchor_col)
                & (anchor_col < col_edges[2:])
            )

            valid_windows = (
                valid_rows[:, None]
                & valid_cols[None, :]
            )

            scores = np.where(
                valid_windows,
                scores,
                -np.inf
            )

        best_row, best_col = np.unravel_index(
            np.argmax(scores),
            scores.shape
        )

        r0 = row_edges[best_row]
        r1 = row_edges[best_row + 2]

        c0 = col_edges[best_col]
        c1 = col_edges[best_col + 2]

        history.append((r0, r1, c0, c1))

    return {
        "I_roi": I[r0:r1, c0:c1],
        "x_roi": x[c0:c1],
        "y_roi": y[r0:r1],
        "history": history,
        "final_bounds": (r0, r1, c0, c1)
    }

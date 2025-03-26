from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


def calibrate_py(p_y: "NDArray[float64]", p_cf: "NDArray[float64]", mode: str = "diagonal") -> "NDArray[float64]":
    if mode not in ["diagonal", "identity"]:
        msg = f"Invalid calibration mode: {mode}. Must be one of: ['diagonal', 'identity']"
        raise ValueError(msg)

    num_classes = p_y.shape[0]

    if mode == "diagonal":
        W = np.linalg.inv(np.identity(num_classes) * p_cf)
        b = np.zeros([num_classes, 1])
        cal_py = np.matmul(W, np.expand_dims(p_y, axis=-1)) + b

    elif mode == "identity":
        W = np.identity(num_classes)
        b = -1 * np.expand_dims(np.log(p_cf), axis=-1)
        cal_py = np.matmul(W, np.expand_dims(np.log(p_y + 10e-6), axis=-1)) + b
        cal_py = np.exp(cal_py)

    return cal_py / np.sum(cal_py)

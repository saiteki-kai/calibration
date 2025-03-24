from typing import Literal

import numpy as np
import numpy.typing as npt


def calibrate_py(
    p_y: npt.NDArray[np.float64],
    p_cf: npt.NDArray[np.float64] | None,
    mode: Literal["diagonal", "identity"] = "diagonal",
) -> npt.NDArray[np.float64]:
    num_classes = p_y.shape[0]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
            cal_py = np.matmul(W, np.expand_dims(p_y, axis=-1)) + b
        elif mode == "identity":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(np.log(p_cf), axis=-1)
            cal_py = np.matmul(W, np.expand_dims(np.log(p_y + 10e-6), axis=-1)) + b
            cal_py = np.exp(cal_py)

    cal_py = cal_py / np.sum(cal_py)

    return cal_py

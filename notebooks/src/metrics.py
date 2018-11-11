import numpy as np


def total_log_loss(y_predict, y_true):
    y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)
    y_true = np.clip(y_true, 1e-15, 1 - 1e-15)

    log_pred = np.log(y_predict)
    log_pred_1 = np.log(np.subtract(1, y_predict))

    y_true_1 = np.subtract(1, y_true)

    log_values = np.add(np.multiply(y_true, log_pred),
                        np.multiply(y_true_1, log_pred_1))
    return - np.mean(log_values, dtype=np.float64)

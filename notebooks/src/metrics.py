def log_loss(predict, y_value):
    predict = max(min(predict, 1. - 10e-15), 10e-15)
    return -log(predict) if y_value == 1. else -log(1. - predict)

def total_log_loss(y_predict, y_true):
    loss = 0
    quantity = 0
    for pred, y_value in zip(y_predict, y_true):
        c += 1
        loss += logloss(pred, y_true)
    return loss / c
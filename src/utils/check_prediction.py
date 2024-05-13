def predict(true_pred, pred):
    if pred >= 0.9 * true_pred and pred <= 1.1 * true_pred:
        return True
    else:
        return False


def custom_accuracy_score(y_test, y_pred) -> float:
    counter = 0
    results = []
    total_len = len(y_pred)
    for true_pred, pred in zip(y_test, y_pred):
        pred_valid = predict(true_pred, pred)
        if pred_valid:
            counter += 1
        results.append(pred_valid)
    return float(counter / total_len), results

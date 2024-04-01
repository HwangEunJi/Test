import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))



def _naive_forecasting(true, seasonality = 1):
    """ Naive forecasting method which just repeats previous samples """
    return true[:-seasonality]

def MASE(pred, true, true_train, seasonality = 1):
    """ Mean Absolute Scaled Error """
    return MAE(pred, true) / MAE(_naive_forecasting(true_train, seasonality), true_train[seasonality:])

def RMSSE(pred, true, true_train, seasonality = 1):
    """ Root Mean Squared Scaled Error """
    return RMSE(pred, true) / RMSE(_naive_forecasting(true_train, seasonality), true_train[seasonality:])



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def metric_check(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return print('MAE: ', mae, ', ', 'MSE: ', mse, ', ', 'RMSE: ', rmse, ', ', 'MAPE: ', mape, ', ', 'MSPE: ', mspe)


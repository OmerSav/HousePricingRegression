from IPython.display import HTML, display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def html_table(df):
    if (isinstance(df, pd.Series)):
        df = df.to_frame()
    display(HTML(df.to_html()))


def residuals_calc(X, y, model):
    return (y - model.predict(X))


def rmsle(y_true, y_pred):
    return np.sqrt(
        np.mean(
            np.square(np.log(y_pred + 1) - np.log(y_true + 1))))


def residual_plot(y_true, residuals, title, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    ax.scatter(y_true, residuals)
    ax.axhline(linestyle='dashed', c='y')
    ax.set(title=title, ylabel='Residuals',
           xlabel='Y True Values')


def residual_plots(model, X_train, X_test, y_train, y_test):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[12, 8], dpi=100)

    fig.tight_layout(pad=5.0)

    residuals_train = residuals_calc(X_train, y_train, model)
    residuals_test = residuals_calc(X_test, y_test, model)
    residual_plot(y_train, residuals_train, 'Train', axs[0, 0])
    sns.kdeplot(residuals_train, ax=axs[0, 1])
    axs[0, 1].set(title='Train')
    residual_plot(y_test, residuals_test, 'Test', axs[1, 0])
    sns.kdeplot(residuals_test, ax=axs[1, 1])
    axs[1, 1].set(title='Test')
    plt.show()


def reg_score_table(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    if y_train_pred.ndim > 1:
        y_train_pred = y_train_pred.flatten()
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    rmsle_train = rmsle(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    if y_test_pred.ndim > 1:
        y_test_pred = y_test_pred.flatten()
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    rmsle_test = rmsle(y_test, y_test_pred)

    return pd.DataFrame(
        {'Train': [r2_train, mae_train, mse_train, rmse_train, rmsle_train],
         'Test': [r2_test, mae_test, mse_test, rmse_test, rmsle_test]},
        index=['R2', 'MAE', 'MSE', 'RMSE', 'RMSLE'])

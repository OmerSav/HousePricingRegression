from IPython.display import HTML, display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def html_table(df):
    if(isinstance(df, pd.Series)):
        df = df.to_frame()
    display(HTML(df.to_html()))

def residuals_calc(X, y, model):
    return (y - model.predict(X))

def residual_plot(X, y, model, title=''):
    residuals = residuals_calc(X, y, model)
    fig, ax = plt.subplots()
    ax.scatter(y, residuals)
    ax.axhline(linestyle='dashed', c='y')
    ax.set(title=title, ylabel='Residuals',
           xlabel='Y True Values')
    plt.show()
    

def evaluate_regression(model, X_train, X_test, y_train, y_test):
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    print(f'''Evaluations for Sale Prices\n\
    \tR2 Train: {r2_train}\n\
    \tR2 Test: {r2_test}\n\
    \tMean:{y_train.mean()}\n\
    \tMean Absolute Error: {mae}\n\
    \tMean Squared Error: {mse}\n\
    \tRoot Mean Squared Error: {rmse}''')
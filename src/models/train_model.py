import os
from pathlib import Path
import logging
from joblib import dump
import tensorflow as tf
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
import xgboost as xgb

import pandas as pd
from dotenv import find_dotenv, load_dotenv


def main():
    logger = logging.getLogger(__name__)
    logger.info('Training models...')
    processed_data_dir = Path('./data/processed/')
    file_name = 'train.csv'
    file_path = processed_data_dir / file_name
    models_trained_dir = Path('./models/trained/')
    if not os.path.exists(models_trained_dir):
        os.mkdir(models_trained_dir)
    featurebuild_dir = Path('./models/featurebuild/')
    if not os.path.exists(featurebuild_dir):
        os.mkdir(featurebuild_dir)

    df = pd.read_csv(file_path)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    polynomial_converter = PolynomialFeatures(degree=2)
    X_scaled_poly = polynomial_converter.fit_transform(X_scaled)

    def train_deploy(model_class, X, model_name, **kwargs):
        model = model_class(**kwargs)
        model.fit(X, y)
        model_path = models_trained_dir / f'{model_name}.joblib'
        dump(model, model_path)
        return model

    reg_final = train_deploy(LinearRegression, X, '0.1-linearregression')
    polyreg_final = train_deploy(LinearRegression, X_scaled_poly,
                                 '0.2-polynomialregression')
    lasso_final = train_deploy(Lasso, X_scaled, '0.3-lasso',
                               **{'alpha': 100.0})
    ridge_final = train_deploy(Ridge, X_scaled, '0.4-ridge',
                               **{'alpha': 100.0})
    elastic_final = train_deploy(ElasticNet, X_scaled, '0.5-elasticnet',
                                 **{'alpha': 50, 'l1_ratio': 0.99})
    svr_final = train_deploy(SVR, X_scaled, '0.6-supportvector')
    tree_final = train_deploy(DecisionTreeRegressor, X, '0.7-decisiontree',
                              **{'criterion': 'absolute_error', 'max_depth': 5,
                                 'min_impurity_decrease': 1.0,
                                 'min_samples_split': 10})
    rfr_final = train_deploy(RandomForestRegressor, X, '0.8-randomforest',
                             **{'criterion': 'squared_error', 'max_depth': 300,
                                'min_impurity_decrease': 1.0,
                                'n_estimators': 200})
    abr_final = train_deploy(AdaBoostRegressor, X, '0.9-adaboost',
                             **{'learning_rate': 1.0, 'loss': 'square',
                                'n_estimators': 1000})
    gbr_final = train_deploy(GradientBoostingRegressor, X,
                             '0.10-gradientboosting',
                             **{'max_depth': 3,
                                'min_impurity_decrease': 0.21544,
                                'n_estimators': 200})
    xgbr_final = train_deploy(xgb.XGBRegressor, X,
                              '0.11-extremegradientboosting',
                              **{'max_depth': 3, 'n_estimators': 300})

    def nodereg(nodes):
        return tf.keras.layers.Dense(nodes,
                                     kernel_regularizer=tf.keras.regularizers.L1L2(
                                         l1=0.99), )

    def ann_model_init():
        inputs = tf.keras.Input(shape=(X_scaled.shape[1]))

        x1 = nodereg(10)(inputs)
        x = nodereg(20)(x1)
        x = nodereg(20)(x)
        x = nodereg(30)(x)
        x = nodereg(30)(x)
        x = nodereg(20)(x)
        x = nodereg(20)(x)
        x2 = nodereg(10)(x)
        x = tf.concat([x1, x2], axis=1)
        outputs = nodereg(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    ann_final = ann_model_init()

    ann_final.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss='mse')

    def scheduler(epoch, lr):
        if 0 < epoch and epoch % 70 == 0:
            return lr / 2
        else:
            return lr

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    ann_final.fit(x=X_scaled, y=y,
                  epochs=100,
                  batch_size=X_scaled.shape[0],
                  verbose=0,
                  callbacks=[callback, ],
                  shuffle=True)
    ann_final.save(models_trained_dir / f'0.12-neuralnet.h5')

    dump(final_scaler, featurebuild_dir / '0.2-standardscaler.joblib')
    dump(polynomial_converter,
         featurebuild_dir / '0.3-polynomialconverter.joblib')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()

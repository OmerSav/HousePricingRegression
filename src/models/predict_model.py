import os
import click
import logging
from pathlib import Path
import joblib
import datetime

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.utils import build_features

model_names = ['regression', 'polynomial-regression', 'lasso', 'ridge',
               'elasticnet', 'SVR', 'decision-tree', 'ada-boost',
               'random-forest',
               'gradient-boosting', 'XGBR']


@click.command()
@click.option('--model', help='Model name to use.')
@click.option('--data_path', help='Data path to predict.',
              type=click.Path(exists=True))
def main(model, data_path):
    """ Transform new data and predict on it with choosed model. On make file
     config default model is 'ridge' and default data path is
     './data/raw/test.csv'
    ridge
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f'Making prediction on; \'{data_path}\' with model: \'{model}\'')

    if model not in model_names:
        raise Exception(
            f'\nWrong model name \'{model}\'. Please use one of the '
            f'fallowings:\n '
            + ''.join(
                [f'\t{name}\n' for name in model_names]))
    data_path = Path(data_path)
    model_index = model_names.index(model)
    models_path = Path('./models/trained/')
    feature_build_path = Path('./models/featurebuild/')
    models_filenames = os.listdir(models_path)
    models_filenames.sort(
        key=lambda name: int(name.split('-')[0].split('.')[1]))
    mdl = models_filenames[model_index]
    mdl = joblib.load(models_path / mdl)
    enc = joblib.load(feature_build_path / '0.1-onehotencoder.joblib')
    df = pd.read_csv(data_path)
    df_final = build_features(df, enc)
    # num cols after transformation 304
    if df_final.shape[1] != 304:
        raise Exception(logger.info(f'Inputted data shape wrong!'))
    sc = joblib.load(feature_build_path / '0.2-standardscaler.joblib')
    X = sc.transform(df_final)
    if model == 'polynomial-regression':
        pol_c = joblib.load(feature_build_path /
                            '0.3-polynomialconverter.joblib')
        X = pol_c.transform(X)
    predictions = mdl.predict(X)
    time_var = str(datetime.datetime.now()).replace(' ', '_').replace(':',
                                                                      '_').replace(
        '.', '_')
    if not os.path.exists('./data/predictions/'):
        os.mkdir('./data/predictions/')
    saved_file_path = Path(
        f'./data/predictions/{data_path.name.split(".")[0]}_{time_var}.csv')
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(saved_file_path, index=False)

    logger.info(
        f'Prediction completed succesfully see file; '
        f'{saved_file_path.absolute()}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

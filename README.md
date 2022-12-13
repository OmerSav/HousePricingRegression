House Pricing Regression
=================

The project task
from "[Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)"
.
With 79 explanatory variables, we try to estimate house prices. Detailed
information from Kaggle is provided below.

### Data Description from Kaggle

> The Ames Housing dataset was compiled by Dean De Cock for use in data science
> education. It's an incredible alternative for data scientists looking for a
> modernized and expanded version of the often cited Boston Housing dataset.
>
> Ask a home buyer to describe their dream house, and they probably won't begin
> with the height of the basement ceiling or the proximity to an east-west
> railroad. But this playground competition's dataset proves that much more
> influences price negotiations than the number of bedrooms or a white-picket
> fence.
>
>With 79 explanatory variables describing (almost) every aspect of residential
> homes in Ames, Iowa, this competition challenges you to predict the final
> price
> of each home.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train_model`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── predictions    <- Model predictions saved here.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

# **House Pricing Regression - Advanced Regression Techniques Project**

I am working on this task for practicing and demonstrating my expertise in Data
Science.
My main purpose for this project is to show my data science skills without
revealing my client's projects.
I didn't spend time to make this project perfect, and because of that, there can
be mistakes. It is for demonstration purposes without ethical problems.

## Project Structure Explanation

I mainly worked and made every operation on jupyter notebooks to show and
explain ecery step I did,
So you can basically check **notebooks** folder to see them also you can find
pdf versions of these notebooks with same name in
**reports** folder.
Also you can find helper function on **src/utils.py** and data preparation
training, prediction
codes in **src**. This scripts coded for using easily on command line with *
make*
commmads (you can see detailed command prompt api from below).

## Make Api

### *make* data

It generate handled and encoded data from raw data for training. There is
already **train.csv** data to convert but
if you wan't to train on different data with same structer and column names,
simply put data in **data/raw/** folder with naming **train.csv**.

### *make* train_model

**Prerequests**

Proccessed training data.(make data command)

**Usage**

It trains model with **data/proccesed/train.csv** and then saves trained models
to **models/trained/** folder.
It also saves standart scaler and polynomial converter to **
models/featurebuild/**.

### *make* predict

**Prerequests**

Trained models.(make train_model command)

**Usage**

It takes to argument *model*, *data*

> make predict model='elasticnet' data='.../somedatafolder/data.csv'

It predict data and saves result as csv to **data/predictions/** folder with
name of
original file and time stampl fallowed it.

# Training Models

In this section we will train models to predict house prices. To strech fistly I
tried to fir simle OLS Regression both with applying formula and with libraries(
statmodels and seaborn).

After that I tried every model as default and with grid search I reported all
results after every model. Lets look at file structer.

| Model                                            | Library - Framework |
| :----------------------------------------------- | :-----------------: |
| Multiple Regression                              |    scikit-learn    |
| Polinomial Regression                            |    scikit-learn    |
| Lasso Regression (l1 regularization)             |    scikit-learn    |
| Ridge Regression (l2 regularization)             |    scikit-learn    |
| ElasticNet Regression (l1 and l2 regularization) |    scikit-learn    |
| Support Vector Regression                        |    scikit-learn    |
| Decision Tree                                    |    scikit-learn    |
| Random Forest                                    |    scikit-learn    |
| Ada Boost                                        |    scikit-learn    |
| Gradient Boosting                                |    scikit-learn    |
| eXtreme Gradient Boosting                        |       XGBoost       |
| Neural Network                                   |     TensorFlow     |

## Load & Prepare Data

In models data will used as both scaled and unscaled. To scale data I used
standart scaler which calculates every feature Z score with training data set
statistics.

**Load Prepare Data**

* **Data prepation - Trains Test Split and Feature Scaling**

**Simple OLS**

* **Custom Calculation**
* **Calculation with Libraries**

**Regression Models**

* **Multiple Linear Regression Models**
* **Polynomial Regression Model**
* **Grid Search on Linear Regression Models with Regularization**
    * **Lasso**
    * **Ridge**
    * **ElasticNet**

&nbsp;**Support Vector Regression**

&nbsp;**Tree Based Models**

* &nbsp;&nbsp;**Decision Tree Regressor**
* &nbsp;&nbsp;**Random Forest Regressor**
* &nbsp;&nbsp;**AdaBoost Regressor**
* &nbsp;&nbsp;**Gradient Boosting Regressor**
* &nbsp;&nbsp;**eXtreme Gradient Boosting**

**Neural Network**

**Deploying Models**

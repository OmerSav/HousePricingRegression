HousePricesKaggle
=================

The project task
from "[Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)"
.

Project Organization
--------------------

├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
-------------------------------------------------------------------------------------------

# **House Prices - Advanced Regression Techniques Project**

The project task
from "[Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)"
.
I am working on this task for the practicing and demostration of my expertise on
Data Science.
My main purpose for this project is to show my data science skills without
reveal my clients projects.
I didn't spent time to make this project perfect. It is just my regular
performance for the demostration
purpose without etical problems.

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

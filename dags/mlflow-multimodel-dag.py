from airflow.decorators import task, dag, task_group
from airflow.utils.task_group import TaskGroup
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook

from datetime import datetime

import logging
import mlflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


docs = """
MLFlow:
Airflow can integrate with tools like MLFlow to streamline the model experimentation process. By using the automation and orchestration of Airflow together with MLflow's core concepts (Tracking, Projects, Models, and Registry) Data Scientists can standardize, share, and iterate over experiments more easily.


XCOM Backend:
By default, Airflow stores all return values in XCom. However, this can introduce complexity, as users then have to consider the size of data they are returning. Futhermore, since XComs are stored in the Airflow database by default, intermediary data is not easily accessible by external systems.
By using an external XCom backend, users can easily push and pull all intermediary data generated in their DAG in GCS.
"""


def log_roc_curve(y_test: list, y_pred: list):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr,tpr) 
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()


def log_confusion_matrix(y_test: list, y_pred: list):
    cm = confusion_matrix(y_test, y_pred)
    t_n, f_p, f_n, t_p = cm.ravel()
    mlflow.log_metrics({'True Positive': t_p, 'True Negative': t_n, 'False Positive': f_p, 'False Negatives': f_n})

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()


def log_classification_report(y_test: list, y_pred: list):
    cr = classification_report(y_test, y_pred, output_dict=True)
    logging.info(cr)
    cr_metrics = pd.json_normalize(cr, sep='_').to_dict(orient='records')[0]
    mlflow.log_metrics(cr_metrics)


def log_all_eval_metrics(y_test: list, y_pred: list):
    
    # Classification Report
    log_classification_report(y_test, y_pred)


    # Confusion Matrix
    log_confusion_matrix(y_test, y_pred)


    # ROC Curve
    log_roc_curve(y_test, y_pred)


def test(clf, test_set, model_type):    
    logging.info('Gathering Validation set results')
    y_pred = clf.predict(test_set)

    if model_type == 'lgbm':
        return np.where(y_pred > 0.5, 1, 0)
    else:
        return y_pred

@dag(
    start_date=datetime(2021, 1, 1),
    schedule_interval=None,
    catchup=False,
    doc_md=docs
)
def mlflow_multimodel_example():

    @task
    def load_data():
        """Pull Census data from Public BigQuery and save as Pandas dataframe in GCS bucket with XCom"""

        bq = BigQueryHook()
        sql = """
        SELECT * FROM `bigquery-public-data.ml_datasets.census_adult_income`
        """

        return bq.get_pandas_df(sql=sql, dialect='standard')



    @task
    def preprocessing(df: pd.DataFrame):
        """Clean Data and prepare for feature engineering
        
        Returns pandas dataframe via Xcom to GCS bucket.

        Keyword arguments:
        df -- Raw data pulled from BigQuery to be processed. 
        """

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Clean Categorical Variables (strings)
        cols = df.columns
        for col in cols:
            if df.dtypes[col]=='object':
                df[col] =df[col].apply(lambda x: x.rstrip().lstrip())


        # Rename up '?' values as 'Unknown'
        df['workclass'] = df['workclass'].apply(lambda x: 'Unknown' if x == '?' else x)
        df['occupation'] = df['occupation'].apply(lambda x: 'Unknown' if x == '?' else x)
        df['native_country'] = df['native_country'].apply(lambda x: 'Unknown' if x == '?' else x)


        # Drop Extra/Unused Columns
        df.drop(columns=['education_num', 'relationship', 'functional_weight'], inplace=True)

        return df

    @task
    def feature_engineering(df: pd.DataFrame):
        """Feature engineering step
        
        Returns pandas dataframe via XCom to GCS bucket.

        Keyword arguments:
        df -- data from previous step pulled from BigQuery to be processed. 
        """

        
        # Onehot encoding 
        df = pd.get_dummies(df, prefix='workclass', columns=['workclass'])
        df = pd.get_dummies(df, prefix='education', columns=['education'])
        df = pd.get_dummies(df, prefix='occupation', columns=['occupation'])
        df = pd.get_dummies(df, prefix='race', columns=['race'])
        df = pd.get_dummies(df, prefix='sex', columns=['sex'])
        df = pd.get_dummies(df, prefix='income_bracket', columns=['income_bracket'])
        df = pd.get_dummies(df, prefix='native_country', columns=['native_country'])


        # Bin Ages
        df['age_bins'] = pd.cut(x=df['age'], bins=[16,29,39,49,59,100], labels=[1, 2, 3, 4, 5])


        # Dependent Variable
        df['never_married'] = df['marital_status'].apply(lambda x: 1 if x == 'Never-married' else 0) 


        # Drop redundant colulmn
        df.drop(columns=['income_bracket_<=50K', 'marital_status', 'age'], inplace=True)

        return df


    @task_group(group_id='grid_search_cv')
    def grid_search_cv(features):
        """Train and validate model
        
        Returns accuracy score via XCom to GCS bucket.

        Keyword arguments:
        df -- data from previous step pulled from BigQuery to be processed. 
        """

        model_grid_params = { 
            'lgbm':{
                'learning_rate': [0.01, .05, .1], 
                'n_estimators': [50, 100, 150],
                'num_leaves': [31, 40, 80],
                'max_depth': [16, 24, 31, 40],
                'boosting_type': ['gbdt'], 
                'objective': ['binary'],
                'seed': [55]
                },
            'log_reg':{
                'penalty': ['l1','l2','elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
            }


        models = [
            ['lgbm', lgb.LGBMClassifier(objective='binary', metric=['auc', 'binary_logloss'])],
            ['log_reg', LogisticRegression(max_iter=500)]
        ]

        tasks = []

        for model in models:
            @task(task_id=model[0])
            def train(df: pd.DataFrame, model_type=model[0],model=model[1], grid_params=model_grid_params[model[0]], **kwargs):

                y = df['never_married']
                X = df.drop(columns=['never_married'])


                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55, stratify=y)
                
                mlflow.set_tracking_uri('http://host.docker.internal:5000')
                try:
                    # Creating an experiment 
                    mlflow.create_experiment('census_prediction')
                except:
                    pass
                # Setting the environment with the created experiment
                mlflow.set_experiment('census_prediction')

                grid_search = GridSearchCV(model, param_grid=grid_params, verbose=1, cv=5, n_jobs=-1)


                mlflow.sklearn.autolog()

                with mlflow.start_run(run_name=f'{model_type}_{kwargs["run_id"]}'):

                    logging.info('Performing Gridsearch')
                    grid_search.fit(X_train, y_train)

                    logging.info(f'Best Parameters\n{grid_search.best_params_}')
                    best_params = grid_search.best_params_


                    if model_type == 'lgbm':

                        mlflow.sklearn.autolog()

                        train_set = lgb.Dataset(X_train, label=y_train)
                        test_set = lgb.Dataset(X_test, label=y_test)

                        best_params['metric'] = ['auc', 'binary_logloss']

                        logging.info('Training model with best parameters')
                        clf = lgb.train(
                            train_set=train_set,
                            valid_sets=[train_set, test_set],
                            valid_names=['train', 'validation'],
                            params=best_params,
                            early_stopping_rounds=5
                        )
                        
                    else:
                        logging.info('Training model with best parameters')
                        clf = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], solver=best_params['solver']).fit(X_train, y_train)

                    y_pred_class = test(clf, X_test, model_type)

                    # Log Classfication Report, Confustion Matrix, and ROC Curve
                    log_all_eval_metrics(y_test, y_pred_class)

            tasks.append(train(features))
        return tasks

    df = load_data()
    clean_data = preprocessing(df)
    features = feature_engineering(clean_data)
    grid_search_cv(features)

    
dag = mlflow_multimodel_example()

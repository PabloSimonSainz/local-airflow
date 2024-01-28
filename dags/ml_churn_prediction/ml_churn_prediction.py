from datetime import datetime
import pickle
import json
from random import randint

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from ml_churn_prediction.preprocessor.i_preprocessor import IPreprocessor
from ml_churn_prediction.preprocessor.churn_preprocessor import ChurnPreprocessor
from ml_churn_prediction.trainer.i_trainer import ITrainer
from ml_churn_prediction.trainer.churn_trainer import ChurnTrainer

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'customer_id'.lower()
Y_COL = 'churn'.lower()
TABLE_NAME = 'churn_prediction_dataset'

MODELS_PATH = f"/opt/airflow/data/models"
CONFIG_PATH = f"/opt/airflow/config"

def _get_data() -> pd.DataFrame:
    """
    Read dataset from Postgres
    """
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    print(f"Reading data from {TABLE_NAME}...")
    
    df:pd.DataFrame = pd.read_sql(f"SELECT * FROM {TABLE_NAME}_preprocessed", engine)
    
    return df

def validate(**context) -> None:
    """
    Evaluate model, y is a binary variable
    
    :param split_seed: seed used to split dataset
    :type split_seed: int
    
    :return: None
    """
    model_data = context['ti'].xcom_pull(key='training_result')
    
    model_name = model_data['model_name']
    seed = model_data['seed']
    
    eval_table_name = f"{TABLE_NAME}_evaluation"
    
    df = _get_data()
    
    # train test split
    y = df[Y_COL]
    X = df.drop([Y_COL, PK], axis=1)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # load model
    model_path = f"{MODELS_PATH}/{model_name}.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # evaluate model
    y_pred = model.predict(X_test)
    
    # evaluate model with the whole dataset
    y_pred_all = model.predict(X)
    
    # insert id, accuracy, precision, recall, f1 score to Postgres
    values = {
        "id":[model_name],
        "test_accuracy":[np.mean(y_pred==y_test)],
        "test_precision":[np.mean(y_pred[y_test==1]==y_test[y_test==1])],
        "test_recall":[np.mean(y_pred[y_test==1]==y_test[y_test==1])],
        "test_f1_score":[np.mean(y_pred[y_test==1]==y_test[y_test==1])],
        "test_true_positive":[np.sum(y_pred[y_test==1]==1)],
        "test_true_negative":[np.sum(y_pred[y_test==0]==0)],
        "test_false_positive":[np.sum(y_pred[y_test==0]==1)],
        "test_false_negative":[np.sum(y_pred[y_test==1]==0)],
        "all_accuracy":[np.mean(y_pred_all==y)],
        "all_precision":[np.mean(y_pred_all[y==1]==y[y==1])],
        "all_recall":[np.mean(y_pred_all[y==1]==y[y==1])],
        "all_f1_score":[np.mean(y_pred_all[y==1]==y[y==1])],
        "all_true_positive":[np.sum(y_pred_all[y==1]==1)],
        "all_true_negative":[np.sum(y_pred_all[y==0]==0)],
        "all_false_positive":[np.sum(y_pred_all[y==0]==1)],
        "all_false_negative":[np.sum(y_pred_all[y==1]==0)]
    }
    
    id:str = model_name
    
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    df = pd.DataFrame(values)
    
    df.to_sql(eval_table_name, engine, index=False, if_exists='append')
    
with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
    # PREPROCESSOR
    preprocessor:IPreprocessor = ChurnPreprocessor(
        pk=PK,
        y=Y_COL,
        conn_id=ID_CONNECTION,
        table_name=TABLE_NAME
    )
    
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocessor.preprocess
    )
    
    # TRAINER
    trainer:ITrainer = ChurnTrainer(
        pk=PK,
        y=Y_COL,
        conn_id=ID_CONNECTION,
        table_name=TABLE_NAME
    )
        
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=trainer.train,
        provide_context=True
    )
    
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=validate,
        provide_context=True
    )
    
    preprocess_data >> train_model >> evaluate_model
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

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'customer_id'.lower()
Y_COL = 'churn'.lower()
TABLE_NAME = 'churn_prediction_dataset'

MODELS_PATH = f"/opt/airflow/data/models"
CONFIG_PATH = f"/opt/airflow/config"

def _read_config() -> dict:
    """
    Read config from Postgres
    """
    with open(f"{CONFIG_PATH}/{DAG_ID}.json", 'r') as f:
        config = json.load(f)
        
    return config

def _get_data() -> pd.DataFrame:
    """
    Read dataset from Postgres
    """
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    print(f"Reading data from {TABLE_NAME}...")
    
    df:pd.DataFrame = pd.read_sql(f"SELECT * FROM {TABLE_NAME}_preprocessed", engine)
    
    return df

def train() -> dict:
    """
    Train model
    """
    seed = randint(0, 100)
    
    df = _get_data()
    
    # sort by PK ensures split with same seed will produce same result
    df = df.sort_values(by=PK)
    
    # train test split
    y = df[Y_COL]
    X = df.drop([Y_COL, PK], axis=1)
    
    for i in X.columns:
        print(f"{i}: TYPE={X[i].dtype}, MAX={X[i].max()}, MIN={X[i].min()}, HAS_NULL={X[i].isnull().sum()}")
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # build model using pipeline
    model = Pipeline(steps=[
        ('classifier', RandomForestClassifier())
    ])
    
    model.fit(X_train, y_train)
    
    # save model
    model_name = f"model_{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    model_path = f"{MODELS_PATH}/{model_name}"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return {
        "seed":seed, 
        "model_name":model_name
    }

def validate(seed:int, model_name:str) -> None:
    """
    Evaluate model, y is a binary variable
    
    :param split_seed: seed used to split dataset
    :type split_seed: int
    
    :return: None
    """
    eval_table_name = f"{TABLE_NAME}_evaluation"
    
    df = _get_data()
    
    # train test split
    y = df[Y_COL]
    X = df.drop([Y_COL, PK], axis=1)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # load model
    model_path = f"{MODELS_PATH}/{model_name}"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # evaluate model
    y_pred = model.predict(X_test)
    
    # insert id, accuracy, precision, recall, f1 score to Postgres
    values = {
        "id":[model_name],
        "accuracy":[np.mean(y_pred==y_test)],
        "precision":[np.mean(y_pred[y_test==1]==y_test[y_test==1])],
        "recall":[np.mean(y_pred[y_test==1]==y_test[y_test==1])],
        "f1_score":[np.mean(y_pred[y_test==1]==y_test[y_test==1])],
        "true_positive":[np.sum(y_pred[y_test==1]==1)],
        "true_negative":[np.sum(y_pred[y_test==0]==0)],
        "false_positive":[np.sum(y_pred[y_test==0]==1)],
        "false_negative":[np.sum(y_pred[y_test==1]==0)]
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
        
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train
    )
    
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=validate,
        op_kwargs={'seed': train_model.output['seed'], 'model_name': train_model.output['model_name']}
    )
    
    preprocess_data >> train_model >> evaluate_model
    #train_model >> evaluate_model
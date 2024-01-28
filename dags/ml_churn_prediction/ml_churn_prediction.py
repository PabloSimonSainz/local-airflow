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
    select_cols:list = _read_config()['features']
    select_cols = set([PK, Y_COL] + select_cols)
    select_cols = [i.lower() for i in select_cols]
    select_cols = list(set(select_cols))
    
    select_cols:str = ', '.join(select_cols)
    
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    print(f"Reading data from {TABLE_NAME}...")
    
    df:pd.DataFrame = pd.read_sql(f"SELECT {select_cols} FROM {TABLE_NAME}", engine)
    
    return df

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='IQR', factor=1.5):
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        if self.method == 'IQR':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - self.factor * IQR
            self.upper_bound = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        X_clipped = X.clip(lower=self.lower_bound, upper=self.upper_bound, axis=1)
        return X_clipped

def _build_preprocessor(X: pd.DataFrame, y: pd.Series) -> ColumnTransformer:
    """
    Build enhanced preprocessor pipeline
    """
    config = _read_config()
    
    cols = [i.lower() for i in config["features"]]
    
    # numerical nan
    fill_na_mean_cols = [i.lower() for i in config["preprocess"]["missing_values"]["mean"]]
    fill_na_median_cols = [i.lower() for i in config["preprocess"]["missing_values"]["median"]]
        
    #outlier_cols = config["preprocess"]["outliers"]
    
    categorical_features = list(X.select_dtypes(include='object').columns)
    numerical_features = list(X.select_dtypes(exclude='object').columns)
    
    default_na_cols = list(set(numerical_features) - set(fill_na_mean_cols) - set(fill_na_median_cols))
    
    # check that numerical nan columns are numerical
    assert len(set(fill_na_mean_cols + fill_na_median_cols) - set(numerical_features)) == 0, "Can not be non-numerical columns in mean or median config"
    
    print(f'Categorical features({len(categorical_features)}): {categorical_features}')
    print(f'Numerical features({len(numerical_features)}): {numerical_features}')
    
    # build pipeline
    categorical_pipeline = Pipeline(steps=[
        ('fill_na', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_pipeline = Pipeline(steps=[
        ('outlier', OutlierHandler()),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('fill_na_mean', SimpleImputer(strategy='mean'), fill_na_mean_cols),
        ('fill_na_median', SimpleImputer(strategy='median'), fill_na_median_cols),
        ('fill_na_numerical_default', SimpleImputer(strategy='constant', fill_value=0), default_na_cols),
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

def preprocess_data() -> None:
    """
    Enhanced preprocess dataset and save to Postgres
    """
    df = _get_data()
    
    for col in df.columns:
        if "U" in df[col].unique():
            print(f"{col} ({df[col].dtype}): {df[col].unique()}")
    
    # preprocess data
    y = df[Y_COL]
    X = df
    
    for col in X.columns:
        print(f"{col}: {X[col].dtype}")
    
    preprocessor = _build_preprocessor(X=X, y=y)
    
    X = preprocessor.fit_transform(X=X, y=y)
    
    # save preprocessed data
    df = pd.DataFrame(X)
    df[Y_COL] = y
    
    pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
    engine = create_engine(pg_hook.get_uri())
    
    df.to_sql(f"{TABLE_NAME}_preprocessed", engine, index=False, if_exists='replace')
    
def train() -> dict:
    """
    Train model
    """
    seed = randint(0, 100)
    
    df = _get_data()
    
    # train test split
    y = df[Y_COL]
    X = df.drop([Y_COL, PK], axis=1)
    
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
    
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
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
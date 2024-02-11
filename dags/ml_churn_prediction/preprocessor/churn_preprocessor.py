from datetime import datetime
import pickle
import json
from random import randint

from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from ml_churn_prediction.preprocessor.pipe_utils.outlier_handler import OutlierHandler
from ml_churn_prediction.preprocessor.i_preprocessor import IPreprocessor


DAG_ID = 'ml_churn_prediction'

CONFIG_PATH = f"/opt/airflow/config"

def _read_config() -> dict:
    """
    Read config from Postgres
    """
    with open(f"{CONFIG_PATH}/{DAG_ID}.json", 'r') as f:
        config = json.load(f)
        
    return config

class ChurnPreprocessor(IPreprocessor):
    def __init__(self, pk, y, conn_id, table_name):
        self._pk = pk.lower()
        self._y = y.lower()
        self._postgres_conn_id = conn_id
        self._table_name = table_name
    
    def _get_data(self) -> pd.DataFrame:
        """
        Read dataset from Postgres
        """
        select_cols:list = _read_config()['features']
        select_cols = set([self._pk, self._y] + select_cols)
        select_cols = [i.lower() for i in select_cols]
        select_cols = list(set(select_cols))
        
        select_cols:str = ', '.join(select_cols)
        
        pg_hook = PostgresHook(postgres_conn_id=self._postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        print(f"Reading data from {self._table_name}...")
        
        print(f"Selecting columns: {select_cols}")
        df:pd.DataFrame = pd.read_sql(f"SELECT {select_cols} FROM {self._table_name}", engine)
        print(f"Columns: {df.columns}")
        
        return df

    @staticmethod
    def _build_preprocessor(X: pd.DataFrame, y: pd.Series) -> ColumnTransformer:
        """
        Build enhanced preprocessor pipeline
        """
        def_nan_num_strategy='mean'
        
        config = _read_config()
        
        numerical_cols = [i.lower() for i in X.select_dtypes(exclude=['object']).columns]
        categorical_cols = [i.lower() for i in X.select_dtypes(include=['object']).columns]
        
        # missing values
        fill_na_mean_cols = [i.lower() for i in config["preprocess"]["missing_values"].get("mean")]
        fill_na_median_cols = [i.lower() for i in config["preprocess"]["missing_values"].get("median")]
        fill_na_mode_cols = [i.lower() for i in config["preprocess"]["missing_values"].get("mode")]
        
        # numerical outliers
        outlier_cols = [i.lower() for i in config["preprocess"]["outliers"]]
        
        # numerical PIPELINE
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy=def_nan_num_strategy)),
            ('scaler', StandardScaler())
        ])
        
        # categorical PIPELINE
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # preprocessor
        preprocessor = ColumnTransformer([
            ('mean', SimpleImputer(strategy='mean'), fill_na_mean_cols),
            ('median', SimpleImputer(strategy='median'), fill_na_median_cols),
            ('mode', SimpleImputer(strategy='most_frequent'), fill_na_mode_cols),
            ('outlier_handler', OutlierHandler(), outlier_cols),
            ('num', num_pipe, numerical_cols),
            ('cat', cat_pipe, categorical_cols)
        ], remainder='passthrough')
        
        return preprocessor

    def preprocess(self) -> None:
        """
        Enhanced preprocess dataset and save to Postgres
        """
        df = self._get_data()
        
        # lower case column names
        df.columns = df.columns.str.lower()
        
        # preprocess data
        pk = df[self._pk]
        y = df[self._y]
        X = df.drop([self._pk, self._y], axis=1).copy()
        
        preprocessor = ChurnPreprocessor._build_preprocessor(X=X, y=y)
        
        X = preprocessor.fit_transform(X=X, y=y)
        
        # cast to float32
        X = X.astype(np.float32)
        
        # save preprocessed data
        df = pd.DataFrame(X)
        
        df[self._y] = y
        df[self._pk] = pk
        
        # search any columns with "missing" value and replace with np.nan, then drop them
        df = df.replace('missing', np.nan)
        df = df.dropna(axis=1, how='any')
        
        pg_hook = PostgresHook(postgres_conn_id=self._postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        df.to_sql(f"{self._table_name}_preprocessed", engine, if_exists='replace', index=False)
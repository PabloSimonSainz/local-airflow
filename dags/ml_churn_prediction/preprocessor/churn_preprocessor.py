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

MODELS_PATH = f"/opt/airflow/data/models"
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
        self._pk = pk
        self._y = y
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
        
        df:pd.DataFrame = pd.read_sql(f"SELECT {select_cols} FROM {self._table_name}", engine)
        
        return df

    @staticmethod
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

    def preprocess(self) -> None:
        """
        Enhanced preprocess dataset and save to Postgres
        """
        df = self._get_data()
        
        # preprocess data
        y = df[self._y]
        X = df
        
        for col in X.columns:
            print(f"{col}: {X[col].dtype}")
        
        preprocessor = ChurnPreprocessor._build_preprocessor(X=X, y=y)
        
        X = preprocessor.fit_transform(X=X, y=y)
        
        # save preprocessed data
        df = pd.DataFrame(X)
        df[self._y] = y
        
        pg_hook = PostgresHook(postgres_conn_id=self._postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        df.to_sql(f"{self._table_name}_preprocessed", engine, index=False, if_exists='replace')
        
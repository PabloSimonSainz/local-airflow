from functools import reduce

import os
import json

import sqlalchemy

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from preprocessor.i_preprocessor import IPreprocessor

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'Customer_ID'
Y_COL = 'churn'
TABLE_NAME = 'churn_prediction_dataset'

CONFIG_NAME = "churn_prediction.json"
MODELS_PATH = f"/opt/airflow/data/models"

CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"

class Preprocessor(IPreprocessor):
    def _read_config() -> dict:
        """
        Read config from Postgres
        """
        actual_path = os.path.dirname(os.path.realpath(__file__))
        # back 3 folders
        root_path = reduce(lambda x, _: os.path.dirname(x), range(3), actual_path)
        config_path = os.path.join(root_path, 'config', CONFIG_NAME)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config

    def _get_data() -> pd.DataFrame:
        """
        Read dataset from Postgres
        """
        select_cols:list = Preprocessor._read_config()['features']
        select_cols.append(Y_COL)
        
        select_cols:str = ', '.join(select_cols)
        
        engine = sqlalchemy.create_engine(CONN)
        df = pd.read_sql(f"SELECT {select_cols} FROM {TABLE_NAME} ORDER BY {PK}", engine)
                
        return df

    def _remove_outliers(X:pd.DataFrame) -> tuple:
        """
        """
        features = Preprocessor._read_config()['preprocess']['outliers']
        df = X.copy()
        
        indices = [x for x in df.index]    
        out_indexlist = []
            
        for col in features:       
            Q1 = np.nanpercentile(df[col], 25.)
            Q3 = np.nanpercentile(df[col], 75.)
            
            cut_off = (Q3 - Q1) * 1.5
            upper, lower = Q3 + cut_off, Q1 - cut_off
                    
            outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
            out_indexlist.extend(outliers_index)
            
        #using set to remove duplicates
        out_indexlist = list(set(out_indexlist))
        
        clean_data = np.setdiff1d(indices,out_indexlist)

        return X.loc[clean_data]

    def _fill_missing_values(X:pd.DataFrame) -> pd.DataFrame:
        """
        """
        config = Preprocessor._read_config()['preprocess']['missing_values']
        df = X.copy()
        
        for col in df.columns:
            if col in config['mean']:
                df[col].fillna(df[col].mean(), inplace=True)
            elif col in config['literal']:
                df[col].fillna(config['literal'][col], inplace=True)
            
        return df

    def _build_preprocessor(df:pd.DataFrame) -> Pipeline:
        """
        Build preprocessor pipeline
        """
        cathegorical_features = list(df.select_dtypes(include='object').columns)
        numerical_features = list(df.select_dtypes(exclude='object').columns)
        
        print(f'Categorical features({len(cathegorical_features)}): {cathegorical_features}')
        print(f'Numerical features({len(numerical_features)}): {numerical_features}')
        
        # missing values
        missing_values_transformer = Pipeline(steps=[
            ('fillNaN', FunctionTransformer(Preprocessor._fill_missing_values))
        ])
        
        # outliers
        outliers_transformer = Pipeline(steps=[
            ('removeOutliers', FunctionTransformer(Preprocessor._remove_outliers))
        ])
        
        # transformers
        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # column transformer
        preprocessor = ColumnTransformer(transformers=[
            missing_values_transformer,
            outliers_transformer,
            ('cat', cat_transformer, cathegorical_features),
            ('num', num_transformer, numerical_features)
        ])
        
        return preprocessor

    @staticmethod
    def preprocess_data() -> None:
        """
        Preprocess dataset and save to Postgres
        
        :return: None
        """
        df = Preprocessor._get_data()
        
        preprocessor = Preprocessor._build_preprocessor(df)
        
        X = df.drop(Y_COL, axis=1)
        y = df[Y_COL]
        
        X = preprocessor.fit_transform(X)
        
        df = pd.DataFrame(X, columns=df.drop(Y_COL, axis=1).columns)
        df[Y_COL] = y
        
        engine = sqlalchemy.create_engine(CONN)
        
        df.to_sql(f"{TABLE_NAME}_gold", engine, index=False, if_exists='replace')

from functools import reduce
import pickle

import os
import json

import sqlalchemy

from sklearn.model_selection import train_test_split

import pandas as pd

from validator.i_validator import IValidator

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'Customer_ID'
Y_COL = 'churn'
TABLE_NAME = 'churn_prediction_dataset'

CONFIG_NAME = "churn_prediction.json"
MODELS_PATH = f"/opt/airflow/data/models"

CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"

class Validator(IValidator):
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
        select_cols:list = Validator._read_config()['features']
        select_cols.append(Y_COL)
        
        select_cols:str = ', '.join(select_cols)
        
        engine = sqlalchemy.create_engine(CONN)
        df = pd.read_sql(f"SELECT {select_cols} FROM {TABLE_NAME}_gold ORDER BY {PK}", engine)
                
        return df

    @staticmethod
    def validate(seed:int, model_name:str) -> None:
        """
        Evaluate model
        
        :param split_seed: seed used to split dataset
        :type split_seed: int
        
        :return: None
        """
        df = Validator._get_data()
        
        # train test split
        X = df.drop([Y_COL, PK], axis=1)
        y = df[Y_COL]
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # load model
        model_path = f"{MODELS_PATH}/{model_name}"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # evaluate model
        y_pred = model.predict(X_test)
        
        # save results
        results = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred
        })
        
        engine = sqlalchemy.create_engine(CONN)
        
        results.to_sql(f"{TABLE_NAME}_results", engine, index=False, if_exists='replace')
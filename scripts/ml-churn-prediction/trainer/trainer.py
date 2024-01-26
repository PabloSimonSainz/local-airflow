from functools import reduce
from datetime import datetime
import pickle
from random import randint

import os
import json

import sqlalchemy

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from trainer.i_trainer import ITrainer

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'Customer_ID'
Y_COL = 'churn'
TABLE_NAME = 'churn_prediction_dataset'

CONFIG_NAME = "churn_prediction.json"
MODELS_PATH = f"/opt/airflow/data/models"

CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"

class Trainer(ITrainer):
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
        select_cols:list = Trainer._read_config()['features']
        select_cols.append(Y_COL)
        
        select_cols:str = ', '.join(select_cols)
        
        engine = sqlalchemy.create_engine(CONN)
        df = pd.read_sql(f"SELECT {select_cols} FROM {TABLE_NAME}_gold ORDER BY {PK}", engine)
                
        return df

    @staticmethod
    def train_model() -> dict:
        """
        Train model
        """
        # Generate seed to ensure the same data split in different modules
        seed = randint(0, 100)
        
        df = Trainer._get_data()
        
        # train test split
        X = df.drop([Y_COL, PK], axis=1)
        y = df[Y_COL]
        
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
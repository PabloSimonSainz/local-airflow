from datetime import datetime
import pickle
from random import randint

from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml_churn_prediction.preprocessor.i_preprocessor import IPreprocessor
from ml_churn_prediction.preprocessor.churn_preprocessor import ChurnPreprocessor

from ml_churn_prediction.trainer.i_trainer import ITrainer

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

MODELS_PATH = f"/opt/airflow/data/models"

class ChurnTrainer(ITrainer):
    def __init__(self, pk, y, conn_id, table_name):
        self._pk = pk
        self._y = y
        self._postgres_conn_id = conn_id
        self._table_name = table_name
    
    def _get_data(self) -> pd.DataFrame:
        """
        Read dataset from Postgres
        """
        pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
        engine = create_engine(pg_hook.get_uri())
        
        print(f"Reading data from {self._table_name}...")
        
        df:pd.DataFrame = pd.read_sql(f"SELECT * FROM {self._table_name}_preprocessed", engine)
        
        return df

    def train(self, **context):
        """
        Train model
        """
        seed = randint(0, 100)
        
        df = self._get_data()
        
        # sort by self._pk ensures split with same seed will produce same result
        df = df.sort_values(by=self._pk)
        
        # train test split
        y = df[self._y]
        X = df.drop([self._y, self._pk], axis=1)
        
        for i in X.columns:
            print(f"{i}: TYPE={X[i].dtype}, MAX={X[i].max()}, MIN={X[i].min()}, HAS_NULL={X[i].isnull().sum()}")
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # build model using pipeline
        model = Pipeline(steps=[
            ('classifier', RandomForestClassifier())
        ])
        
        model.fit(X_train, y_train)
        
        # save model
        model_name = f"model_{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_path = f"{MODELS_PATH}/{model_name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        result = {
            "seed":seed, 
            "model_name":model_name
        }
        
        print(f"Trained model {model_name} with seed {seed}")
        
        # print feature importance
        feature_importance = model['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature':X_train.columns,
            'importance':feature_importance
        })
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        # save feature importance
        feature_importance_path = f"{MODELS_PATH}/{model_name}_feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        
        context['ti'].xcom_push(key='training_result', value=result)

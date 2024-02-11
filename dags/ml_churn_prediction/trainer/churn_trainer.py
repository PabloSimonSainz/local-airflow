from datetime import datetime
import pickle
from random import randint

from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import mlflow
from mlflow.tracking import MlflowClient

from ml_churn_prediction.preprocessor.i_preprocessor import IPreprocessor
from ml_churn_prediction.preprocessor.churn_preprocessor import ChurnPreprocessor

from ml_churn_prediction.trainer.i_trainer import ITrainer

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

class ChurnTrainer(ITrainer):
    def __init__(self, pk:str, y:str, conn_id:str, table_name:str, models_path:str, experiment_data:dict):
        """
        Constructor
        
        :param pk: primary key
        :param y: Target column
        :param conn_id: Connection id
        :param table_name: Dataset table name
        :param models_path: Models path
        :param tracking_uri: Mlflow tracking uri
        """
        self._pk = pk
        self._y = y
        self._postgres_conn_id = conn_id
        self._models_path = models_path
        self._table_name = table_name
        
        if 'experiment_name' not in experiment_data:
            raise ValueError("experiment_name is required in experiment_data")
        
        self._experiment_data = experiment_data
    
    def _get_data(self) -> pd.DataFrame:
        """
        Read dataset from Postgres
        """
        pg_hook = PostgresHook(postgres_conn_id=ID_CONNECTION)
        engine = create_engine(pg_hook.get_uri())
        
        print(f"Reading data from {self._table_name}...")
        
        df:pd.DataFrame = pd.read_sql(f"SELECT * FROM {self._table_name}_preprocessed", engine)
        
        return df
    
    def _create_model(self, model, params) -> Pipeline:
        """
        Create model
        
        :param seed: Random seed
        """
        model = Pipeline(steps=[
            ('classifier', model(**params))
        ])
        
        return model

    def train(self, **context):
        """
        Train model
        """
        model = LGBMClassifier
        model_name = model.__name__
        params = {
            "num_leaves": 55,
            "learning_rate": 0.01,
            "max_depth": 27,
            "n_estimators": 850,
            "min_child_samples": 100,
            "min_child_weight": 60,
            "subsample": 0.7,
            "max_bin": 300,
            "cat_smooth": 70,
            "cat_l2": 10,
            "verbosity": -1
        }
        
        seed = randint(0, 100)
        
        df = self._get_data()
        
        # sort by self._pk ensures split with same seed will produce same result
        df = df.sort_values(by=self._pk)
        
        # train test split
        y = df[self._y]
        X = df.drop([self._y, self._pk], axis=1)
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # build model using pipeline
        model = self._create_model(model, params)
        
        with mlflow.start_run() as run:
            mlflow.log_param("seed", seed)
            mlflow.log_param("model_name", model_name)
            
            for k, v in params.items():
                mlflow.log_param(k, v)
            
            start_time = datetime.now()
            model.fit(X_train, y_train)
            end_time = datetime.now()
            
            mlflow.log_metric("training_time", (end_time - start_time).seconds)
            
            # register model
            mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
            
            # get run id
            run_id = run.info.run_id
            
        # save model
        model_name = f"{model_name.lower()}_{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_path = f"{self._models_path}/{model_name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        response = {
            "seed":seed, 
            "model_name":model_name,
            "run_id":run_id
        }
        
        print(f"Trained model {model_name} with seed {seed}")
        
        # push result to xcom
        context['ti'].xcom_push(
            key='training_result', 
            value=response
        )

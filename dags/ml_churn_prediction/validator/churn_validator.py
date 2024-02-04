from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from ml_churn_prediction.validator.i_validator import IValidator

MODELS_PATH = f"/opt/airflow/data/models"
CONFIG_PATH = f"/opt/airflow/config"

class ChurnValidator(IValidator):
    def __init__(self, pk, y, conn_id, table_name):
        self._pk = pk
        self._y = y
        self._postgres_conn_id = conn_id
        self._table_name = table_name
    
    def _get_data(self) -> pd.DataFrame:
        """
        Read dataset from Postgres
        """
        pg_hook = PostgresHook(postgres_conn_id=self._postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        print(f"Reading data from {self._table_name}...")
        
        df:pd.DataFrame = pd.read_sql(f"SELECT * FROM {self._table_name}_preprocessed", engine)
        
        return df

    def validate(self, **context) -> None:
        """
        Evaluate model, y is a binary variable
        
        :param split_seed: seed used to split dataset
        :type split_seed: int
        
        :return: None
        """
        model_data = context['ti'].xcom_pull(key='training_result')
        
        model_name = model_data['model_name']
        seed = model_data['seed']
        
        eval_table_name = f"{self._table_name}_evaluation"
        
        df = self._get_data()
        
        # sort by self._pk ensures split with same seed will produce same result
        df = df.sort_values(by=self._pk)
        
        # train test split
        y = df[self._y]
        X = df.drop([self._y, self._pk], axis=1)
        
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
        
        pg_hook = PostgresHook(postgres_conn_id=self._postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        df = pd.DataFrame(values)
        
        df.to_sql(eval_table_name, engine, index=False, if_exists='append')
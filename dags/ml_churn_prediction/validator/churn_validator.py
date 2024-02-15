import os
from airflow.providers.postgres.hooks.postgres import PostgresHook

from sqlalchemy import create_engine
import pickle
from datetime import datetime
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from airflow.hooks.S3_hook import S3Hook

from ml_churn_prediction.validator.i_validator import IValidator

MODELS_PATH = f"/opt/airflow/data/models"
CONFIG_PATH = f"/opt/airflow/config"

POSTGRES_CONN_ID = os.environ.get('POSTGRES_CONN_ID', 'postgres_conn')
S3_CONN_ID = os.environ.get('S3_CONN_ID', 's3_conn')

MIN_ACCURACY = 0.6

class ChurnValidator(IValidator):
    def __init__(self, pk:str, y:str, table_name:str="churn_prediction", 
                 bucket_name:str="bucket", key:str="output/models"):
        """
        Constructor
        
        :param pk: primary key
        :param y: Target column
        :param table_name: Dataset table name
        :param postgres_conn_id: Postgres connection id
        :param s3_conn_id: S3 connection id
        """
        self._pk = pk
        self._y = y
        self._table_name = table_name
        
        self._bucket_name = bucket_name
        self._key = key
        
        self._postgres_conn_id = POSTGRES_CONN_ID
        self._s3_conn_id = S3_CONN_ID
    
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
        run_id = model_data['run_id']
        
        eval_table_name = f"{self._table_name}_evaluation"
        
        df = self._get_data()
        
        # sort by self._pk ensures split with same seed will produce same result
        df = df.sort_values(by=self._pk)
        
        # train test split
        y = df[self._y]
        X = df.drop([self._y, self._pk], axis=1)
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        test_shape = X_test.shape
        
        # load model
        s3 = S3Hook(aws_conn_id=self._s3_conn_id)
        
        response = s3.get_key(
            key=f"{self._key}/{self._table_name}/{model_name}.pkl",
            bucket_name=self._bucket_name
        )
        body = response.get()['Body'].read()
        
        model = pickle.loads(body)
        
        # evaluate model
        start_time = datetime.now()
        y_pred = model.predict(X_test)
        end_time = datetime.now()
        
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
        
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metric("accuracy", values["test_accuracy"][0])
            mlflow.log_metric("precision", values["test_precision"][0])
            mlflow.log_metric("recall", values["test_recall"][0])
            mlflow.log_metric("f1_score", values["test_f1_score"][0])
            
            mlflow.log_param("test_shape", test_shape)
            mlflow.log_metric("evaluation_time", (end_time - start_time).seconds)
            
            acc_threshold = context['params'].get('acc_threshold', MIN_ACCURACY)
            
            if values["test_accuracy"][0] >= acc_threshold:
                s3_path = f"s3://{self._bucket_name}/{self._key}/{model_name}.pkl"
                #mlflow.log_artifact(s3_path, "model")
        
        pg_hook = PostgresHook(postgres_conn_id=self._postgres_conn_id)
        engine = create_engine(pg_hook.get_uri())
        
        df = pd.DataFrame(values)
        
        df.to_sql(eval_table_name, engine, index=False, if_exists='append')
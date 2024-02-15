import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

import mlflow

from ml_churn_prediction.preprocessor.i_preprocessor import IPreprocessor
from ml_churn_prediction.preprocessor.churn_preprocessor import ChurnPreprocessor
from ml_churn_prediction.trainer.i_trainer import ITrainer
from ml_churn_prediction.trainer.churn_trainer import ChurnTrainer
from ml_churn_prediction.validator.i_validator import IValidator
from ml_churn_prediction.validator.churn_validator import ChurnValidator

ID_CONNECTION = os.environ.get('POSTGRES_CONN_ID', 'postgres_conn')
DAG_ID = 'ml_churn_prediction'

PK = 'customer_id'.lower()
Y_COL = 'churn'.lower()

CONFIG_PATH = f"/opt/airflow/config"

EXPERIMENT_DATA = {
    'experiment_name':'churn_prediction',
    'tracking_uri':'http://host.docker.internal:5000'
}

TABLE_NAME = 'churn_prediction'
KEY = "output/models"
BUCKET_NAME = "airflow"

    
with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    mlflow.set_tracking_uri(EXPERIMENT_DATA['tracking_uri'])
    try:
        mlflow.create_experiment(EXPERIMENT_DATA['experiment_name'], artifact_location=f"s3://{BUCKET_NAME}/mlflow/{TABLE_NAME}")
    except Exception as e:
        mlflow.set_experiment(EXPERIMENT_DATA['experiment_name'])
        
    # PREPROCESSOR
    preprocessor:IPreprocessor = ChurnPreprocessor(
        pk=PK,
        y=Y_COL,
        conn_id=ID_CONNECTION,
        table_name=TABLE_NAME
    )
    
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocessor.preprocess
    )
    
    # TRAINER
    trainer:ITrainer = ChurnTrainer(
        pk=PK,
        y=Y_COL,
        experiment_data=EXPERIMENT_DATA,
        table_name=TABLE_NAME,
        bucket_name=BUCKET_NAME,
        key=KEY
    )
        
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=trainer.train,
        provide_context=True
    )
    
    # VALIDATOR
    validator:IValidator = ChurnValidator(
        pk=PK,
        y=Y_COL,
        table_name=TABLE_NAME,
        key=KEY,
        bucket_name=BUCKET_NAME,
    )
    
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=validator.validate,
        provide_context=True
    )
    
    preprocess_data >> train_model >> evaluate_model
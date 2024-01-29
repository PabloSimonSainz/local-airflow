from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from ml_churn_prediction.preprocessor.i_preprocessor import IPreprocessor
from ml_churn_prediction.preprocessor.churn_preprocessor import ChurnPreprocessor
from ml_churn_prediction.trainer.i_trainer import ITrainer
from ml_churn_prediction.trainer.churn_trainer import ChurnTrainer
from ml_churn_prediction.validator.i_validator import IValidator
from ml_churn_prediction.validator.churn_validator import ChurnValidator

ID_CONNECTION = 'postgres_conn'
DAG_ID = 'ml_churn_prediction'

PK = 'customer_id'.lower()
Y_COL = 'churn'.lower()
TABLE_NAME = 'churn_prediction_dataset'

MODELS_PATH = f"/opt/airflow/data/models"
CONFIG_PATH = f"/opt/airflow/config"
    
with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
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
        conn_id=ID_CONNECTION,
        table_name=TABLE_NAME
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
        conn_id=ID_CONNECTION,
        table_name=TABLE_NAME
    )
    
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=validator.validate,
        provide_context=True
    )
    
    preprocess_data >> train_model >> evaluate_model
import pandas as pd

from datetime import datetime

import os

import sys


from etl_local_to_posgres.ingestor.ingestor_csv_to_postgres import IngestorCSVToPostgres
from etl_local_to_posgres.ingestor.i_ingestor import IIngestor

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

DAG_ID = 'etl_local_to_posgres'
POSTGRE_CONN_ID = 'postgres_conn'
TABLE_NAME = 'churn_prediction_dataset'
DATASET_PATH = f"/opt/airflow/data/raw/churn_prediction_dataset.csv"

PK='Customer_ID'

    
etl_process:IIngestor = IngestorCSVToPostgres(
    postgres_conn_id='postgres_conn',
    table_name='churn_prediction_dataset',
    dataset_path="/opt/airflow/data/raw/churn_prediction_dataset.csv",
    pk='Customer_ID'
)

with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
    t1 = PythonOperator(
        task_id='bulk_csv_to_postgres',
        python_callable=etl_process.bulk_csv_to_postgres
    )
    
    t2 = PythonOperator(
        task_id='upsert_data_to_postgres',
        python_callable=etl_process.upsert_data_to_postgres,
        op_kwargs={'cols': t1.output}
    )
    
    t1 >> t2
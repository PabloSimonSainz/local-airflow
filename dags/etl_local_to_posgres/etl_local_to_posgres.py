import pandas as pd

from datetime import datetime

import os

import sys


from etl_local_to_posgres.ingestor.ingestor_csv_to_postgres import IngestorCSVToPostgres
from etl_local_to_posgres.ingestor.i_ingestor import IIngestor

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

DAG_ID = 'etl_local_to_posgres'

POSTGRES_CONN_ID = os.environ.get('POSTGRES_CONN_ID', 'postgres_conn')
    
etl_process:IIngestor = IngestorCSVToPostgres(
    postgres_conn_id=POSTGRES_CONN_ID,
    table_name='churn_prediction',
    dataset_path="/opt/airflow/data/raw/churn_prediction.csv",
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
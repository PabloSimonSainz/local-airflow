from airflow import DAG
from airflow.operators.postgres_operator import PostgresOperator
from datetime import datetime

import os

DAG_ID = 'execute_dataset_ddl'
POSTGRE_CONN_ID = 'postgres_conn'

DATASET_NAME = 'churn_prediction_dataset.sql'
DATASET_PATH = f"/opt/airflow/ddls"

with DAG(
    dag_id=DAG_ID,
    #schedule_interval="@once",
    start_date=datetime.now(),
    template_searchpath=[DATASET_PATH]
    ):
    
    execute_ddls = PostgresOperator(
        task_id='execute_ddls',
        postgres_conn_id=POSTGRE_CONN_ID,
        sql=DATASET_NAME,
        autocommit=True
    )
    
    execute_ddls
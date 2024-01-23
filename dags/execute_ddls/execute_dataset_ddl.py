# Script for executing DDLs in PosgreSQL with Airflow

from airflow import DAG
from airflow.operators.python_operator import PostgresOperator
from datetime import datetime

import os

DAG_ID = 'execute_dataset_ddl'
POSTGRE_CONN_ID = 'postgres_conn'

AUX_PATH = os.path.dirname(os.path.realpath(__file__))
# return 2 times
for i in range(2):
    AUX_PATH = os.path.dirname(AUX_PATH)
SQL_PATH = os.path.join(AUX_PATH, 'ddls', 'dataset.sql')

with DAG(
    dag_id=DAG_ID,
    schedule_interval="@once",
    start_date=datetime.now()
) as dag:
    execute_ddls = PostgresOperator(
        task_id='execute_ddls',
        postgres_conn_id=POSTGRE_CONN_ID,
        sql=SQL_PATH,
        autocommit=True
    )
    
    execute_ddls
# Script that loads data from a local CSV file to PosgreSQL with Airflow

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

import os

DAG_ID = 'etl_local_to_posgres'
POSTGRE_CONN_ID = 'postgres_conn'

# move from airflow/dags/etl_local_to_posgres/etl_local_to_posgres.py to airflow/date/etl_local_to_posgres/etl_local_to_posgres.py
ACTUAL_PATH = os.path.dirname(os.path.realpath(__file__))

    
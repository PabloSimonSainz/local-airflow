from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable

import pandas as pd

from datetime import datetime

from sqlalchemy import create_engine


import os

DAG_ID = 'etl_local_to_posgres'
POSTGRE_CONN_ID = 'postgres_conn'
TABLE_NAME = 'churn_prediction_dataset'
DATASET_PATH = f"/opt/airflow/data/raw/churn_prediction_dataset.csv"

PK='Customer_ID'

def bulk_csv_to_postgres() -> list:
    """
    Copy from csv to postgres table.
    
    :return: list of table columns inserted
    :rtype: list
    """
    # Create connection to postgres
    pg_hook = PostgresHook(postgres_conn_id=POSTGRE_CONN_ID)
    engine = create_engine(pg_hook.get_uri())
    
    # Load data from csv to pandas dataframe
    df = pd.read_csv(DATASET_PATH, sep=';', decimal=',')
    
    # lowercase column names
    df.columns = df.columns.str.lower()
    
    # Upsert data to postgres
    df.to_sql(f"{TABLE_NAME}_temp", engine, index=False, if_exists='replace')
    
    print(f"Upserted data to {TABLE_NAME}_temp table")
    
    return df.columns.tolist()

def upsert_data_to_postgres(cols: list) -> None:
    """
    Upsert data from postgres table to another postgres table.
    """
    # Create connection to postgres
    pg_hook = PostgresHook(postgres_conn_id=POSTGRE_CONN_ID)
    engine = create_engine(pg_hook.get_uri())
    
    # Upsert data to postgres
    with engine.connect() as conn:
        conn.execute(f"""
            INSERT INTO {TABLE_NAME} ({', '.join(cols)})
            SELECT {', '.join(cols)}
            FROM {TABLE_NAME}_temp
            ON CONFLICT ({PK}) DO UPDATE
            SET {', '.join([f"{col}=EXCLUDED.{col}" for col in cols])}
        """)
        
        conn.execute(f"DROP TABLE {TABLE_NAME}_temp")
        
    print(f"Upserted data to {TABLE_NAME} table")
    
    
    
with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
    t1 = PythonOperator(
        task_id='bulk_csv_to_postgres',
        python_callable=bulk_csv_to_postgres
    )
    
    t2 = PythonOperator(
        task_id='upsert_data_to_postgres',
        python_callable=upsert_data_to_postgres,
        op_kwargs={'cols': t1.output}
    )
    
    t1 >> t2
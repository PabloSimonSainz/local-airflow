from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

import pandas as pd
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable
from sqlalchemy import create_engine


import os

DAG_ID = 'etl_local_to_posgres'
POSTGRE_CONN_ID = 'postgres_conn'
DATASET_NAME = 'churn_prediction_dataset'
DATASET_PATH = f"/opt/airflow/data/raw/{DATASET_NAME}.csv"

PK='Customer_ID'

def get_posgres_cols(table_name:str) -> list:
    """
    Get the columns and their types from a postgres table.
    :param table_name: str, Name of the PostgreSQL table
    :return: list, List of column names in the table
    """
    try:
        with PostgresHook(postgres_conn_id=POSTGRE_CONN_ID).get_conn() as conn:
            query = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """
            df = pd.read_sql(query, conn)
            return df['column_name'].tolist()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

def upsert_data_to_postgres():
    print(f'Loading data from {DATASET_PATH} to PostgreSQL...')
    df = pd.read_csv(DATASET_PATH, sep=';', decimal=',')
    print(f'Loaded {df.shape[0]} rows')

    postgres_cols:list = get_posgres_cols(DATASET_NAME)
    
    print(f"Postgres columns: {postgres_cols}")
    
    # Create a connection with Postgres
    postgres_hook = PostgresHook(postgres_conn_id=POSTGRE_CONN_ID)
    with postgres_hook.get_conn() as connection, connection.cursor() as cursor:
        cols = df.columns.tolist()
        cols.pop(cols.index(PK))
        update_set = ', '.join([f'{col}=excluded.{col}' for col in cols])
        
        # df cols lower case
        df.columns = df.columns.str.lower()
        df = df[postgres_cols]
            
        # Upsert the whole DataFrame to a new table, the pk=Customer_ID
        added_rows = 0
        for _, row in df.iterrows():
            if len(row) == 0:
                continue
            
            values = ', '.join([f"'{row[col]}'" if isinstance(row[col], str) else str(row[col]) for col in df.columns])
            try:
                cursor.execute(f"""
                    INSERT INTO {DATASET_NAME} ({', '.join(df.columns)})
                    VALUES ({values})
                    ON CONFLICT ({PK}) DO UPDATE SET {update_set};
                """)
                connection.commit()
                added_rows += 1
            except Exception as e:
                print(f"Error inserting row: {e}")

        print(f'Upserted {added_rows} rows')
    
with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
    load_data_to_postgres = PythonOperator(
        task_id='load_data_to_postgres',
        python_callable=upsert_data_to_postgres
    )

    load_data_to_postgres
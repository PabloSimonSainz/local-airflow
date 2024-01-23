from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime

def test_postgres():
    hook = PostgresHook(postgres_conn_id='postgres_conn')
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT 1')
    result = cursor.fetchone()
    if result[0] == 1:
        print("Connection successful!")
    else:
        print("Connection failed!")

with DAG(
    dag_id='test_postgres', 
    start_date=datetime.now()
    ):

    check_pg_conn_operator = PythonOperator(
        task_id='check_postgres_connection',
        python_callable=test_postgres
    )
    
    check_pg_conn_operator
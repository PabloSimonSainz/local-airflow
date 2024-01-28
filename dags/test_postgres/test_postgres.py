from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime

CON_SUCCESS_MSG = "Connection successful!"
CON_FAILURE_MSG = "Connection failed!"

def test_postgres():
    
    try:
        hook = PostgresHook(postgres_conn_id='postgres_conn')
        conn = hook.get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        
        if result[0] == 1:
            print(CON_SUCCESS_MSG)
        else:
            print(CON_FAILURE_MSG)
    except Exception as e:
        print(e)
        print(CON_FAILURE_MSG)

with DAG(
    dag_id='test_postgres', 
    start_date=datetime.now()
    ):

    check_pg_conn_operator = PythonOperator(
        task_id='check_postgres_connection',
        python_callable=test_postgres
    )
    
    check_pg_conn_operator
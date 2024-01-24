from airflow import DAG
import sqlalchemy as sa
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def test_postgres():
    uri = 'postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
    engine = sa.create_engine(uri)
    
    try:
        conn = engine.connect()
        conn.execute("SELECT 1")
        print("CONNECTION SUCCESSFUL!")
    except Exception as e:
        print(e)
        print("CONNECTION FAILED!")
    finally:
        conn.close()

with DAG(
    dag_id='test_postgres_sa', 
    start_date=datetime.now()):

    test_operator = PythonOperator(
        task_id='test_postgres_sa',
        python_callable=test_postgres
    )

    test_operator
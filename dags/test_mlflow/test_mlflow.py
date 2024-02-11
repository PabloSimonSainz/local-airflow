from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import mlflow
from datetime import datetime

# host.docker.internal
TRACKING_URI = 'http://host.docker.internal:5000'

DAG_ID = 'test_mlflow'
EXPERIMENT_NAME = 'test_mlflow'

def create_mlflow_experiment(experiment_name:str)->None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("foo", 1)
        mlflow.log_metric("foo", 2)
        mlflow.log_metric("foo", 3)

with DAG(
    dag_id=DAG_ID,
    start_date=datetime.now()
    ):
    
    create_mlflow_experiment = PythonOperator(
        task_id='create_mlflow_experiment',
        python_callable=create_mlflow_experiment,
        provide_context=True,
        op_kwargs={'experiment_name':EXPERIMENT_NAME}
    )
    
    create_mlflow_experiment
FROM apache/airflow:2.8.0
COPY requirements.txt .

USER root

# Install libgomp1 for lightgbm
RUN apt-get update && apt-get install -y libgomp1

USER airflow

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
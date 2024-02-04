FROM python:3.7

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

# update pip
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000

CMD [ "mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql://airflow:airflow@postgres/airflow", "--default-artifact-root", "/mlflow/artifacts" ]
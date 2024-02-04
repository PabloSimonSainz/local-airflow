/* Postgresql init */

-- Create airflow database
CREATE DATABASE airflow;

CREATE USER airflow WITH PASSWORD 'airflow';

GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Create db_dev database
CREATE DATABASE db_dev;

CREATE USER developer WITH PASSWORD 'developer';

GRANT ALL PRIVILEGES ON DATABASE db_dev TO developer;

-- Create mlflow database
CREATE DATABASE mlflow;

CREATE USER mlflow WITH PASSWORD 'mlflow';

GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
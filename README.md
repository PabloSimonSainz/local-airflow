# local-airflow

This repository aims to build a robust infrastructure tailored for optimizing machine learning workflows. Leveraging the power of Docker, this setup ensures a seamless and scalable environment for handling various aspects of machine learning projects.

## Core Components

The infrastructure integrates several key open-source services, each chosen for its effectiveness in handling specific facets of machine learning processes:

- **Airflow**: Serves as the backbone for process orchestration. It enables you to programmatically author, schedule, and monitor workflows, ensuring efficient management of complex data pipelines.

- **Postgres**: Acts as our SQL database. Renowned for its reliability and performance, Postgres offers a secure and robust storage solution for managing structured data essential in ML modeling and analysis.

- **MinIO**: Provides a high-performance, AWS S3 compatible object storage system. It's designed for storing and managing large volumes of unstructured data, making it ideal for datasets commonly used in machine learning.

- **MLflow**: A crucial component for monitoring all aspects of the machine learning lifecycle. MLflow tracks experiments, records and compares parameters and results, and assists in model deployment, making it an indispensable tool for maintaining the quality and efficiency of ML models.

With these integrated services, the infrastructure provides a comprehensive platform for managing, tracking, and optimizing machine learning projects from start to finish.

## Project Setup Guide
This guide provides instructions on how to set up the project environment using Docker and establish necessary connections for Postgres and MinIO services.

### Docker Setpu
Docker is used to containerize the application and its services. Follow these steps to build and run your Docker containers:

To build the Docker containers for the project, run the following command in the terminal:
```sh
docker-compose build
```

Once the build process is complete, you can start the containers by running
```sh
docker-compose up -d
```

### Establishing Connections
The project requires connections to Postgres and MinIO services for data storage and management. Use the following commands to set up these connections within the Airflow environment.

Run the following command to add a **Postgres** connection to Airflow:
```sh
docker-compose run airflow-worker connections add postgres_conn `
  --conn-type "postgres" `
  --conn-description "Postgres connection for development environment" `
  --conn-host "postgres" `
  --conn-port 5432 `
  --conn-schema "db_dev" `
  --conn-login "developer" `
  --conn-password "developer"
```

For connecting to a **MinIO** bucket, use the command below:
```sh
docker-compose run airflow-worker connections add s3_conn `
                        --conn-type "aws" `
                        --conn-description "S3 Connection for MinIO bucket" `
                        --conn-extra '{
                            "aws_access_key_id":"minioadmin",
                            "aws_secret_access_key": "minioadmin",
                            "endpoint_url": "http://bucket:9000"
                        }'
```

> **_NOTE:_** The log in credentials depends on the .env file.
FROM python:3.10-slim-buster

EXPOSE 5000

ARG SERVER_DIR=/server

COPY requirements.txt /tmp
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

WORKDIR ${SERVER_DIR}
COPY . ${SERVER_DIR}
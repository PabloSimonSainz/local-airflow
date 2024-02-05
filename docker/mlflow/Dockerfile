FROM python:3.10-slim-buster

WORKDIR ${SERVER_DIR}
COPY . ${SERVER_DIR}


ARG SERVER_DIR=/server

COPY requirements.txt /tmp
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

EXPOSE 5000
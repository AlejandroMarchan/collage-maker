FROM python:3.9-slim

RUN apt-get update && apt-get install -y vim curl jq python-tk

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY app /app

RUN gunicorn --workers=5 --limit-request-line 0 --certfile=/etc/stratio/$SERVICE_NAME.pem --keyfile=/etc/stratio/$SERVICE_NAME.key -b 0.0.0.0:443 app:server
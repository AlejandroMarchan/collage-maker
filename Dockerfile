FROM python:3.9-slim

RUN apt-get update && apt-get install -y vim curl jq python-tk

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY app /app

RUN gunicorn --workers=5 --timeout 120 app:server
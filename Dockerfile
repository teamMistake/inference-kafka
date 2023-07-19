FROM --platform=linux/amd64 python:3.10-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install wget

# COPY ./model_store /app/model_store
COPY ./app /app

CMD ["python","-u","main.py"]

FROM python:3.11-slim-buster

WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv

CMD [ "python3", "app.py" ]



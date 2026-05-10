FROM python:3.8-slim-bullseye
workdir /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
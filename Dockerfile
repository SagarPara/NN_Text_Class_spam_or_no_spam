#getting OS and Python image from DockerHub
FROM python:3.11.9-slim-bullseye

WORKDIR /docker

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#copy all files (including large files)
COPY ./ ./

CMD ["flask", "--app", "predict_if_spam.py", "run", "--host=0.0.0.0", "--port=5001"]


# backend/Dockerfile

FROM python:3.9

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

#COPY fakenews_model /app/fakenews_model

#COPY raw_data /app/raw_data

EXPOSE 8000

CMD ["uvicorn", "fastfakenews:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]
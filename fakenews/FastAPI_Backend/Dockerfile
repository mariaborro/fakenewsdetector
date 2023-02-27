# backend/Dockerfile

FROM python:3.9

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

#COPY fakenews_model /app/fakenews_model

#COPY raw_data /app/raw_data

EXPOSE 8000

#this is the version for deployment. if ran locally we would have "fastfakenews:app" instead of "fakenews.FastAPI_Backend.fastfakenews:app"
CMD ["uvicorn", "fakenews.FastAPI_Backend.fastfakenews:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]
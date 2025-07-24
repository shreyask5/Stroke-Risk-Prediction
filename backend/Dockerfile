FROM python:3.10-slim

WORKDIR /app

RUN test -f .env || echo -e "PORT=5001\nSERVER_IP=127.0.0.1\nSECRET_KEY=your-secret-key-change-in-production\nALLOWED_ORIGINS=*\nMODEL_PATH=./stroke_prediction_model.joblib" > .env

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"] 
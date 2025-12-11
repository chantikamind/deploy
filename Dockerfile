FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:$PORT"]

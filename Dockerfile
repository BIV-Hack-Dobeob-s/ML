FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# input and output directories
VOLUME [ "/app/input", "app/output" ]

CMD ["python", "main.py"]

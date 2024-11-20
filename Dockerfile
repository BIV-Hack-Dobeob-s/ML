FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# installing depedencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# input and output directories
VOLUME [ "/app/input", "app/output" ]

# start up application
CMD ["python", "main.py"]

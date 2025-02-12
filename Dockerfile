FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory
COPY app/ ./app/

# Copy environment file
COPY .env.example .env

EXPOSE 4242

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4242", "--reload"] 
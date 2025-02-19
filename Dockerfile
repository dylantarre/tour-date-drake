FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and bot runner
COPY app app/
COPY bot_runner.py .

EXPOSE 4343

# Use environment variable to determine which service to run
ENV SERVICE=api
CMD if [ "$SERVICE" = "bot" ]; then \
        python bot_runner.py; \
    else \
        uvicorn app.main:app --host 0.0.0.0 --port 4343; \
    fi 
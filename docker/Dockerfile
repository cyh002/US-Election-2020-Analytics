# docker build : docker build -f docker/dockerfile . -t streamlit-app:latest
FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown

# Create data/results directory
RUN mkdir -p data/results

# Make directory
RUN mkdir -p conf src app

# 2. Verify directory structure
RUN ls -la

# Download results data
RUN gdown https://drive.google.com/uc?id=1x7UrY60whtL45mvTl-Z80qxWf84D9T6l -O data/results/results_xlm.csv

# Copy application code
COPY conf/ ./conf/
COPY src/ ./src/
COPY app/ ./app/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Port exposure (adjust as needed)
EXPOSE 8501

# set home directory
WORKDIR /app

# Command to run the application
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    jq \
    libcairo2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Dash app folder
COPY app/ app/

# Set PYTHONPATH so Gunicorn can find app.app
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Use entrypoint for proper signal handling
ENTRYPOINT ["gunicorn"]
CMD ["--workers=5", "--bind=0.0.0.0:8050", "--timeout=120", "app.app:server"]
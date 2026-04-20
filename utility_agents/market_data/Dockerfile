# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# Set the working directory
WORKDIR $APP_HOME

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the requirements file explicitly
COPY requirements.txt .

# Install dependencies using uv into the system site-packages
RUN uv pip install --system -r requirements.txt

# Copy local code to the container image
COPY . .

# Run the web service on container startup.
# Cloud Run automatically injects the PORT environment variable.
CMD ["adk", "api_server", "."]

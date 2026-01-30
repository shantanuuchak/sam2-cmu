FROM python:3.12-slim

# Install system dependencies for OpenCV and SAM2
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system --no-cache .

# Copy the rest of the app
COPY app.py .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
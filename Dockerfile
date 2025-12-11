FROM python:3.10-slim

# Install system dependencies needed by opencv-python-headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (adjust if needed)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]

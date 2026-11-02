FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the ports Render uses
EXPOSE 8000
EXPOSE 8501

# Run the startup script
CMD ["bash", "start.sh"]

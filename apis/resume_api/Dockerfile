# Start from a slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the shared 'libs' package (path from root)
COPY ./libs /app/libs
RUN pip install /app/libs

# Copy this service's requirements and install them (path from root)
COPY ./apis/resume_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy this service's application code (path from root)
COPY ./apis/resume_api /app

# Expose the port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
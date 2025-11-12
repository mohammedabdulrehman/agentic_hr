# Start from a slim Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Copy the shared 'libs' package (path from root)
COPY ./libs /app/libs
RUN pip install /app/libs

# 2. Copy this service's requirements (path from root)
COPY ./workers/aggregator_worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy this service's application code (path from root)
COPY ./workers/aggregator_worker /app


# The command to run the worker
# This will be overridden by the 'command:' in docker-compose.yml
# but it's good practice to have a default.
# We use the variable name from our .env file.
CMD ["celery", "-A", "main", "worker", "--loglevel=info", "-Q", "${QUEUE_AGGREGATOR}"]
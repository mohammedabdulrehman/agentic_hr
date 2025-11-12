# Start from a slim Python image
FROM python:3.11-slim
WORKDIR /app

# Install shared libs
COPY ./libs /app/libs
RUN pip install /app/libs

# Install service-specific requirements
COPY ./workers/parser_worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy service-specific code
COPY ./workers/parser_worker /app

# Command to run (read from docker-compose, but good default)
CMD ["celery", "-A", "main", "worker", "--loglevel=info", "-Q", "${QUEUE_PARSER}"]
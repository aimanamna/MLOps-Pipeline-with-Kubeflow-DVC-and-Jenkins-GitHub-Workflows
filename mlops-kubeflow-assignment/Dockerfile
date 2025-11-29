# Dockerfile - image for running pipeline components (if using custom components/images)
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy src for local execution inside container
COPY src/ /app/src/
ENV PYTHONPATH=/app/src

# entrypoint left generic; components will call functions or scripts as needed
CMD ["bash"]

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create the db directory for the volume mount
RUN mkdir -p /app/db

EXPOSE 8501

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
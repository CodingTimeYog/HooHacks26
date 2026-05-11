FROM python:3.13-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# # Run training pipeline at build time so the image starts instantly
# # 1. Train models 50 times and average results
# RUN python scripts/run_experiments.py
# # 2. Train the farm risk classifier
# RUN python backend/training_and_eval.py
# # 3. Generate forecast cache from trained models
# RUN python backend/run_pipeline.py

# Create the db directory for the volume mount
RUN mkdir -p /app/db

EXPOSE 8501

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
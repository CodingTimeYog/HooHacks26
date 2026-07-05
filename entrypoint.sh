#!/bin/sh
set -e

# When a database is configured, bootstrap it before anything trains:
# land the raw source tables, then build the dbt staging views + mart
# that the feature pipeline reads (train_models errors out if DATABASE_URL
# is set but the mart is missing).
if [ -n "$DATABASE_URL" ]; then
    echo "DATABASE_URL set — bootstrapping raw schema + dbt mart..."
    python backend/src/ingestion/postgres_loader.py
    dbt run --project-dir foregast_dbt --profiles-dir foregast_dbt
fi

# Start the scheduler in the background
python -m backend.scheduler &

# Start Streamlit in the foreground (container stays alive as long as this runs)
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
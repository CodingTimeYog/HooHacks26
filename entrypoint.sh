#!/bin/sh
set -e

# Start the scheduler in the background
python -m backend.scheduler &

# Start Streamlit in the foreground (container stays alive as long as this runs)
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
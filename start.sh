#!/bin/bash
# Start FastAPI backend in the background
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

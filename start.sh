#!/bin/bash
# Start FastAPI backend in the background on port 8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on port 8501 (or port 7860 if using Hugging Face Spaces)
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

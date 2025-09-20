@echo off
echo Starting FastAPI development server...
echo Server will exclude *.log files from hot reload
echo Access the app at: http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.
uv run uvicorn src.api.main:app --reload --reload-exclude="*.log" --host 127.0.0.1 --port 8000

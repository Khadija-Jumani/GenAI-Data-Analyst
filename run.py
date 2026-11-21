"""
run.py
------
One-click entry point to start the GenAI Data Analyst backend server.
Can be executed directly from the project root:
    python run.py
"""

import os
import sys

# Add backend directory to sys.path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🚀 Starting GenAI Data Analyst Server...")
    print(" 📍 Local URL: http://localhost:8000")
    print(" 📚 API Docs:  http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

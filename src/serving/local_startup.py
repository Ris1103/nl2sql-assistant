"""Start local services (Ollama + FastAPI + ngrok) and print the public tunnel URL.

Usage:
    python -m src.serving.local_startup

Requires: pyngrok installed and ngrok authtoken configured.
    ngrok config add-authtoken <your-token>
"""
import subprocess
import sys
import time

import requests
from pyngrok import ngrok

FASTAPI_PORT = 8000
OLLAMA_URL = "http://localhost:11434"


def wait_for(url: str, timeout: int = 30, label: str = "") -> bool:
    for _ in range(timeout):
        try:
            requests.get(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    print(f"  ✗ {label} not ready after {timeout}s")
    return False


def main():
    print("Starting local NL2SQL services...\n")

    # 1. Check / start Ollama
    print("1. Checking Ollama...")
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        print("   Ollama already running")
    except Exception:
        print("   Starting Ollama...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for(f"{OLLAMA_URL}/api/tags", timeout=20, label="Ollama"):
            sys.exit(1)
        print("   Ollama started")

    # 2. Start FastAPI
    print("2. Starting FastAPI...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serving.api:app", "--port", str(FASTAPI_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for(f"http://localhost:{FASTAPI_PORT}/health", timeout=15, label="FastAPI"):
        api_proc.terminate()
        sys.exit(1)
    print(f"   FastAPI running on port {FASTAPI_PORT}")

    # 3. Open ngrok tunnel
    print("3. Opening ngrok tunnel...")
    tunnel = ngrok.connect(FASTAPI_PORT, "http")
    public_url = tunnel.public_url

    print(f"\n{'=' * 52}")
    print(f"  PUBLIC URL:  {public_url}")
    print(f"{'=' * 52}")
    print("\nPaste this URL into the Streamlit UI under Local mode.")
    print("Press Ctrl+C to stop all services.\n")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        ngrok.disconnect(public_url)
        api_proc.terminate()


if __name__ == "__main__":
    main()

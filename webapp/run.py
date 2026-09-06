"""
run.py
======
RetinaGuard Standalone Localhost Web Application Launcher.
Single command to launch the full clinical screening workstation in your browser.

Usage:
    python webapp/run.py
"""

import sys
import time
import socket
import threading
import webbrowser
from pathlib import Path
import uvicorn

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if the given port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def open_browser(url: str, delay_seconds: float = 1.2):
    """Opens browser after slight delay to allow Uvicorn server to bind."""
    time.sleep(delay_seconds)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    # Read host and port from environment with fallback for local and cloud deployment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    # Check port availability if running locally without explicit PORT set in env
    if "PORT" not in os.environ and not is_port_available(port, "127.0.0.1"):
        for candidate_port in range(port + 1, port + 20):
            if is_port_available(candidate_port, "127.0.0.1"):
                port = candidate_port
                break

    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    url = f"http://{display_host}:{port}"

    print("\n" + "=" * 52)
    print("  RETINAGUARD WEB APPLICATION")
    print("=" * 52)
    print(f"\n  Production Model : EXP-001 (EfficientNet-B0 ONNX)")
    print(f"  Host Binding     : {host}")
    print(f"  Port             : {port}")
    print(f"  Access URL       : {url}")
    print("\n  Press Ctrl+C in this terminal to stop the server.")
    print("=" * 52 + "\n")

    # Launch browser thread only in local desktop environment (skip on headless/cloud)
    is_headless = bool(os.environ.get("RENDER") or os.environ.get("CI") or os.environ.get("PORT"))
    if not is_headless:
        print("  Opening browser automatically...\n")
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Start Uvicorn server
    uvicorn.run(
        "webapp.main:app",
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

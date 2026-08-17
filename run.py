import atexit
import asyncio
import os
import shutil
import sys
import threading
import webbrowser
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from core.gesture_engine import GestureEngine

app = FastAPI(title="PIPO ASCII Matrix")

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

gesture_engine = GestureEngine()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await gesture_engine.register(websocket)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        gesture_engine.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

app.mount("/", StaticFiles(directory="public", html=True), name="public")

def cleanup_cache_on_exit():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

atexit.register(cleanup_cache_on_exit)

def start_vision():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    gesture_engine.run_capture(loop)

if __name__ == "__main__":
    print("\n" + "="*55)
    print("      LAUNCHING PIPO WATER-BALL 3D ASCII CORE       ")
    print("="*55 + "\n")

    threading.Thread(target=start_vision, daemon=True).start()
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:8000")).start()

    try:
        uvicorn.run(app, host="localhost", port=8000, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        gesture_engine.running = False
        cleanup_cache_on_exit()
        sys.exit(0)

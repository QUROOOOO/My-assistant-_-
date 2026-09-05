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
from starlette.websockets import WebSocketDisconnect
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

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    gesture_engine.set_server_loop(loop)
    asyncio.create_task(keepalive_telemetry_loop())

@app.on_event("shutdown")
async def on_shutdown():
    gesture_engine.running = False

async def keepalive_telemetry_loop():
    while gesture_engine.running:
        try:
            now = asyncio.get_event_loop().time()
            if now - gesture_engine.last_broadcast_time > 0.05:
                await gesture_engine.broadcast()
        except Exception:
            pass
        await asyncio.sleep(0.033)

@app.websocket("/ws/telemetry")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        await gesture_engine.register(websocket)
    except WebSocketDisconnect:
        pass
    except (ConnectionResetError, RuntimeError):
        pass
    except Exception:
        pass

@app.websocket("/ws/live_feed")
async def live_feed_websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        await gesture_engine.register_live_feed(websocket)
    except WebSocketDisconnect:
        pass
    except (ConnectionResetError, RuntimeError):
        pass
    except Exception:
        pass

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
    gesture_engine.run_capture()

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

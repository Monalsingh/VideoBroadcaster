import uvicorn
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import threading
from stream_utils import Streaming

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

stream_thread = None

streaming = Streaming()

# Set below based on your webcam fps, by default is 15.
# streaming.update_cam_fps(15)


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/start")
def start_stream(
    source: str = Query("0"),
    fps: int = Query(15),
    blur_strength: int = Query(21),
    background: str = Query("none")
):
    streaming.update_streaming_config(in_source=source, out_source=None, fps=fps, blur_strength=blur_strength, background=background)

    global stream_thread

    if streaming.running:
        return JSONResponse(content={"message": "Stream already running"}, status_code=400)

    if fps < 1 or fps > 60:
        return JSONResponse(content={"message": "Invalid FPS value (1-60)"}, status_code=400)

    stream_thread = threading.Thread(
        target=streaming.stream_video, args=()
    )
    stream_thread.start()
    return {"message": f"Streaming started from source: {fps} FPS and blur strength {blur_strength}"}


@app.get("/stop")
def stop_stream():
    streaming.update_running_status()
    return {"message": "Streaming stopped"}


@app.get("/devices")
def devices():
    return streaming.list_available_devices()


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)

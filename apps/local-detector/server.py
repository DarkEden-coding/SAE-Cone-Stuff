"""Local cone-detection webapp backend (FastAPI + YOLO CoreML)."""
import asyncio
import base64
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

import pylon_cam

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
UPLOADS = ROOT / "runs" / "uploads"
VIDEOS = ROOT / "runs" / "videos"
for d in (UPLOADS, VIDEOS):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "models" / "best.mlpackage"
model = YOLO(str(MODEL_PATH), task="detect")
model_lock = threading.Lock()
model(np.zeros((640, 640, 3), np.uint8), verbose=False)  # warm up CoreML
CLASSES = {0: "blue_cone", 1: "yellow_cone"}

app = FastAPI(title="Local Cone Detector")
app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
jobs: dict[str, dict] = {}


def infer(img: np.ndarray, conf: float):
    """Run the model on a BGR image; returns (result, detections, ms)."""
    t0 = time.perf_counter()
    with model_lock:
        result = model(img, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    if result.boxes is not None:
        for c, p, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            c = int(c)
            dets.append({"cls": c, "name": result.names.get(c, CLASSES.get(c, str(c))),
                         "conf": round(p, 4), "box": [round(v, 1) for v in b]})
    return result, dets, round(ms, 1)


@app.get("/")
def index():
    if not (STATIC / "index.html").exists():
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {"classes": {str(k): v for k, v in CLASSES.items()}, "model": MODEL_PATH.name}


@app.post("/api/image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.25)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    result, dets, ms = await asyncio.to_thread(infer, img, conf)
    ok, buf = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 90])
    h, w = img.shape[:2]
    return {"detections": dets, "width": w, "height": h, "ms": ms,
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not open video writer (avc1/mp4v)")


def process_video(job_id: str, src: Path, conf: float):
    job = jobs[job_id]
    cap = cv2.VideoCapture(str(src))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        job["total"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result, dets, _ = infer(frame, conf)
            if writer is None:
                h, w = frame.shape[:2]
                writer = open_writer(VIDEOS / f"{job_id}.mp4", fps, (w, h))
            writer.write(result.plot())
            for d in dets:
                job["counts"][d["name"]] = job["counts"].get(d["name"], 0) + 1
            job["frames"] += 1
            if job["total"]:
                job["progress"] = min(job["frames"] / job["total"], 0.99)
        if job["frames"] == 0:
            raise RuntimeError("No frames could be read from video")
        job["total"] = job["frames"]
        job.update(status="done", progress=1.0)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=str(e))
    finally:
        cap.release()
        if writer is not None:
            writer.release()


@app.post("/api/video")
async def detect_video(file: UploadFile = File(...), conf: float = Form(0.25)):
    job_id = uuid.uuid4().hex[:12]
    src = UPLOADS / f"{job_id}{Path(file.filename or '').suffix or '.mp4'}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"status": "processing", "progress": 0.0, "frames": 0, "total": 0,
                    "counts": {name: 0 for name in CLASSES.values()}, "error": None}
    threading.Thread(target=process_video, args=(job_id, src, conf), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/video/{job_id}")
def video_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


@app.get("/api/video/{job_id}/result")
def video_result(job_id: str):
    path = VIDEOS / f"{job_id}.mp4"
    if jobs.get(job_id, {}).get("status") != "done" or not path.exists():
        raise HTTPException(404, "Result not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.websocket("/ws/stream")
async def stream(ws: WebSocket, conf: float = 0.25):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                await ws.send_json({"error": "Could not decode frame"})
                continue
            _, dets, ms = await asyncio.to_thread(infer, img, conf)
            h, w = img.shape[:2]
            await ws.send_json({"detections": dets, "width": w, "height": h, "ms": ms})
    except WebSocketDisconnect:
        pass


@app.get("/api/pylon")
async def pylon_cameras():
    """Basler cameras reachable through pylon; these are streamed by the server, not the browser."""
    try:
        cameras = await asyncio.to_thread(pylon_cam.list_cameras)
    except Exception as e:  # noqa: BLE001
        return {"available": pylon_cam.pylon is not None, "cameras": [], "error": str(e)}
    return {"available": pylon_cam.pylon is not None, "cameras": cameras}


@app.websocket("/ws/pylon/{serial}")
async def pylon_stream(ws: WebSocket, serial: str, conf: float = 0.25):
    """Server grabs + infers; each frame is sent as a JSON message followed by its JPEG (binary).
    The client acks each rendered frame; at most two frames are in flight.
    Clients may send {"conf": x} to change the threshold without reopening the camera."""
    await ws.accept()
    try:
        camera = await asyncio.to_thread(pylon_cam.PylonCamera, serial)
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"error": f"Could not open camera: {e}"})
        await ws.close()
        return
    state = {"conf": conf}
    credits = asyncio.Semaphore(2)

    def grab_and_infer():
        t0 = time.perf_counter()
        frame = camera.grab()
        _, dets, ms = infer(frame, state["conf"])
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        h, w = frame.shape[:2]
        return {"detections": dets, "width": w, "height": h, "ms": ms,
                "server_ms": round((time.perf_counter() - t0) * 1000, 1)}, jpg.tobytes()

    async def receive():
        while True:
            msg = await ws.receive_json()
            if "conf" in msg:
                state["conf"] = float(msg["conf"])
            if msg.get("ack"):
                credits.release()

    async def send():
        while True:
            await credits.acquire()
            meta, jpg = await asyncio.to_thread(grab_and_infer)
            await ws.send_json(meta)
            await ws.send_bytes(jpg)

    tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            e = t.exception()
            if e is not None and not isinstance(e, WebSocketDisconnect):
                await ws.send_json({"error": f"Camera stream stopped: {e}"})
                await ws.close()
    except Exception:  # noqa: BLE001 - client already gone
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)  # let an in-progress grab finish before closing
        await asyncio.to_thread(camera.close)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

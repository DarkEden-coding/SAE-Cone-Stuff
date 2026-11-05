"""Private cone pose API. Run with ``uv run --locked python -m cone_app.server``."""

import argparse
from contextlib import asynccontextmanager
import hmac
import json
from pathlib import Path
import tempfile
import threading
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from . import __version__, live
from .annotate import render as render_annotated
from .config import Options, Settings
from .intrinsics import ProfileStore, inspect_media, validate_capture
from .jobs import JobBusy, JobStore


def create_app(settings: Settings | None = None, *, store: JobStore | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    temporary = settings.data_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    # Multipart files spill after 1 MiB, on the data disk rather than the home disk.
    tempfile.tempdir = str(temporary.resolve())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = store or JobStore(settings)
        app.state.profiles = ProfileStore(settings.data_dir / "profiles.json")
        app.state.jobs.start()
        try:
            yield
        finally:
            await run_in_threadpool(app.state.jobs.stop)

    app = FastAPI(title="Cone Pose", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        path = request.url.path
        if settings.token and path.startswith("/api/") and path != "/api/health":
            authorization = request.headers.get("authorization", "")
            supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
            if path.startswith("/api/jobs/") and path.rsplit("/", 1)[-1] in ("media", "result", "annotated"):
                supplied = supplied or request.query_params.get("token", "")
            if path.startswith("/api/live/sessions/") and path.endswith("/log"):
                supplied = supplied or request.query_params.get("token", "")
            if not hmac.compare_digest(supplied.encode(), settings.token.encode()):
                return JSONResponse({"detail": "Bearer token required"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        response = await call_next(request)
        if path == "/" or path.startswith("/static/"):
            # Revalidate the UI on every load so frontend updates reach Safari without a hard refresh.
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(KeyError)
    async def not_found(request: Request, error: KeyError):
        return JSONResponse({"detail": "Job not found"}, status_code=404)

    @app.get("/api/health")
    def health():
        import torch

        return {"ok": True, "model": str(settings.weights), "imgsz": settings.imgsz,
                "device": "cuda:0" if torch.cuda.is_available() else "cpu", "version": __version__,
                "auth": bool(settings.token)}

    app.state.live = live.register(app, settings)

    async def spool(file: UploadFile, destination: Path) -> None:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                await run_in_threadpool(output.write, chunk)

    async def upload(request: Request, source: str) -> JSONResponse:
        destination = log_destination = None
        try:
            async with request.form(max_files=2, max_fields=8, max_part_size=64 * 1024 * 1024) as form:
                media_file = form.get("media")
                if not isinstance(media_file, UploadFile):
                    raise ValueError("A media file is required")
                capture_text, profile_id = form.get("capture"), form.get("profile_id")
                if source == "ios" and not capture_text:
                    raise ValueError("capture JSON is required")
                if bool(capture_text) == bool(profile_id):
                    raise ValueError("Provide either capture JSON or profile_id")
                options = Options.model_validate({"imgsz": settings.imgsz,
                                                  **json.loads(str(form.get("options", "{}")))}).model_dump()
                suffix = Path(media_file.filename or "upload").suffix.lower()
                if suffix not in (".jpg", ".jpeg", ".png", ".mp4", ".mov", ".m4v", ".avi", ".webm"):
                    suffix = ".bin"
                destination = temporary / (uuid4().hex + suffix)
                await spool(media_file, destination)
                live_session, live_recording = form.get("live_session_id"), form.get("recording_id")
                live_log = form.get("live_log")
                if live_log is not None or live_session is not None or live_recording is not None:
                    live_session, live_recording = str(live_session or ""), str(live_recording or "")
                    if not (live.valid_id(live_session) and live.valid_id(live_recording)):
                        raise ValueError("live_session_id and recording_id must be valid ids")
                    if isinstance(live_log, UploadFile):
                        log_destination = temporary / (uuid4().hex + ".jsonl")
                        await spool(live_log, log_destination)
                        await run_in_threadpool(app.state.live.check_log, live_session, log_destination)
                metadata = await run_in_threadpool(inspect_media, destination)
                if capture_text:
                    capture = await run_in_threadpool(validate_capture, json.loads(str(capture_text)), metadata)
                    await run_in_threadpool(app.state.profiles.update, capture)
                else:
                    capture = await run_in_threadpool(app.state.profiles.capture, str(profile_id), metadata)
                name = str(form.get("name") or capture.get("name") or media_file.filename or "Untitled capture")
                job = await run_in_threadpool(app.state.jobs.create, destination, capture, metadata, options, name, source)
                if live_session:
                    await run_in_threadpool(app.state.live.link_recording, live_session, live_recording,
                                            job["id"], log_destination)
                return JSONResponse(job, status_code=201)
        except (ValueError, TypeError, ValidationError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            for path in (destination, log_destination):
                if path is not None:
                    path.unlink(missing_ok=True)

    @app.post("/api/captures", status_code=201)
    async def create_capture(request: Request):
        return await upload(request, "ios")

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request):
        return await upload(request, "web")

    @app.get("/api/jobs")
    def list_jobs():
        return app.state.jobs.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return app.state.jobs.get(job_id)

    @app.get("/api/jobs/{job_id}/media")
    def get_media(job_id: str):
        return FileResponse(app.state.jobs.media(job_id))

    render_locks: dict[str, threading.Lock] = {}

    def annotated_file(job_id: str) -> Path:
        job = app.state.jobs.get(job_id)
        directory = app.state.jobs.directory(job_id)
        result_path = directory / "result.json"
        if job["status"] != "done" or not result_path.is_file():
            raise HTTPException(status_code=404, detail="Result is not ready")
        out = directory / ("annotated.mp4" if job["kind"] == "video" else "annotated.jpg")
        with render_locks.setdefault(job_id, threading.Lock()):
            # Re-render after a rerun replaces result.json.
            if not out.is_file() or out.stat().st_mtime < result_path.stat().st_mtime:
                render_annotated(app.state.jobs.media(job_id), json.loads(result_path.read_text()), out)
        return out

    @app.get("/api/jobs/{job_id}/annotated")
    async def get_annotated(job_id: str):
        out = await run_in_threadpool(annotated_file, job_id)
        name = app.state.jobs.get(job_id).get("name") or job_id
        return FileResponse(out, filename=f"{Path(name).stem}-annotated{out.suffix}")

    @app.get("/api/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = app.state.jobs.get(job_id)
        result_path = app.state.jobs.directory(job_id) / "result.json"
        if job["status"] != "done" or not result_path.is_file():
            raise HTTPException(status_code=404, detail="Result is not ready")
        return FileResponse(result_path, media_type="application/json")

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun(job_id: str, request: Request):
        try:
            partial = await request.json()
            if not isinstance(partial, dict):
                raise ValueError("Expected an options object")
            return await run_in_threadpool(app.state.jobs.rerun, job_id, partial)
        except JobBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValidationError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        app.state.jobs.delete(job_id)
        return Response(status_code=204)

    @app.get("/api/intrinsics-profiles")
    def profiles():
        return app.state.profiles.list()

    @app.get("/")
    def index():
        path = settings.static_dir / "index.html"
        return FileResponse(path) if path.is_file() else HTMLResponse("<h1>Cone Pose</h1><p>The API is ready.</p>")

    app.mount("/static", StaticFiles(directory=str(settings.static_dir), check_dir=False), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

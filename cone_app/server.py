"""Private cone pose API. Run with ``uv run --locked python -m cone_app.server``."""

import argparse
from contextlib import asynccontextmanager
import hmac
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from . import __version__
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
            if path.startswith("/api/jobs/") and path.rsplit("/", 1)[-1] in ("media", "result"):
                supplied = supplied or request.query_params.get("token", "")
            if not hmac.compare_digest(supplied.encode(), settings.token.encode()):
                return JSONResponse({"detail": "Bearer token required"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def not_found(request: Request, error: KeyError):
        return JSONResponse({"detail": "Job not found"}, status_code=404)

    @app.get("/api/health")
    def health():
        import torch

        return {"ok": True, "model": str(settings.weights), "imgsz": settings.imgsz,
                "device": "cuda:0" if torch.cuda.is_available() else "cpu", "version": __version__,
                "auth": bool(settings.token)}

    async def upload(request: Request, source: str) -> JSONResponse:
        destination = None
        try:
            async with request.form(max_files=1, max_fields=6, max_part_size=64 * 1024 * 1024) as form:
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
                with destination.open("wb") as output:
                    while chunk := await media_file.read(1024 * 1024):
                        await run_in_threadpool(output.write, chunk)
                metadata = await run_in_threadpool(inspect_media, destination)
                if capture_text:
                    capture = await run_in_threadpool(validate_capture, json.loads(str(capture_text)), metadata)
                    await run_in_threadpool(app.state.profiles.update, capture)
                else:
                    capture = await run_in_threadpool(app.state.profiles.capture, str(profile_id), metadata)
                name = str(form.get("name") or capture.get("name") or media_file.filename or "Untitled capture")
                job = await run_in_threadpool(app.state.jobs.create, destination, capture, metadata, options, name, source)
                return JSONResponse(job, status_code=201)
        except (ValueError, TypeError, ValidationError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            if destination is not None:
                destination.unlink(missing_ok=True)

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

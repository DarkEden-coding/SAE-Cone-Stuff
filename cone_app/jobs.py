"""Atomic on-disk job records and a single sequential inference worker."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import queue
import shutil
import threading
from typing import Callable
from uuid import uuid4

from .config import Options, Settings
from .intrinsics import now, write_json

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


class JobBusy(ValueError):
    pass


class JobStore:
    def __init__(self, settings: Settings, *, runner: Callable | None = None,
                 detector_factory: Callable | None = None):
        self.settings = settings
        self.root = settings.data_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records: dict[str, dict] = {}
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.stopping = threading.Event()
        self.runner = runner
        self.detector_factory = detector_factory
        self.worker: threading.Thread | None = None
        for path in sorted(self.root.glob("*/job.json")):
            try:
                record = json.loads(path.read_text())
                if record["id"] != path.parent.name:
                    raise ValueError("Job directory and id differ")
                if record["status"] in ("queued", "running"):
                    record.update(status="queued", stage="queued", progress=0.0, error=None, updated_at=now())
                    write_json(path, record)
                    self.queue.put(record["id"])
                self.records[record["id"]] = record
            except (ValueError, KeyError, OSError):
                log.exception("Skipping unreadable job record %s", path)

    def start(self) -> None:
        if self.worker is None:
            self.worker = threading.Thread(target=self._work, name="cone-gpu-worker", daemon=True)
            self.worker.start()

    def stop(self) -> None:
        self.stopping.set()
        self.queue.put(None)
        if self.worker is not None:
            self.worker.join(timeout=30)

    def get(self, job_id: str) -> dict:
        with self.lock:
            return deepcopy(self.records[job_id])

    def list(self) -> list[dict]:
        with self.lock:
            return deepcopy(sorted(self.records.values(), key=lambda job: job["created_at"], reverse=True))

    def directory(self, job_id: str) -> Path:
        self.get(job_id)  # IDs are resolved through records, never as arbitrary paths.
        return self.root / job_id

    def media(self, job_id: str) -> Path:
        return next(self.directory(job_id).glob("media.*"))

    def create(self, upload: Path, capture: dict, media: dict, options: dict, name: str, source: str) -> dict:
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
        directory = self.root / job_id
        directory.mkdir()
        timestamp = now()
        record = {"id": job_id, "name": name, "kind": media["kind"], "status": "queued", "stage": "queued",
                  "progress": 0.0, "error": None, "created_at": timestamp, "updated_at": timestamp,
                  "options": options, "image_size": media["image_size"], "n_frames": media["n_frames"],
                  "duration_s": media["duration_s"], "media_url": f"/api/jobs/{job_id}/media",
                  "result_url": None, "source": source}
        try:
            upload.replace(directory / ("media" + upload.suffix))
            write_json(directory / "capture.json", capture)
            write_json(directory / "job.json", record)
            with self.lock:
                self.records[job_id] = record
                self.queue.put(job_id)
            return deepcopy(record)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def rerun(self, job_id: str, partial_options: dict) -> dict:
        with self.lock:
            record = self.records[job_id]
            if record["status"] in ("queued", "running"):
                raise JobBusy("Wait for the active job to finish before rerunning")
            options = Options.model_validate({**record["options"], **partial_options}).model_dump()
            (self.root / job_id / "result.json").unlink(missing_ok=True)
            self._update(job_id, options=options, status="queued", stage="queued", progress=0.0,
                         error=None, result_url=None)
            self.queue.put(job_id)
            return self.get(job_id)

    def delete(self, job_id: str) -> None:
        with self.lock:
            if job_id not in self.records:
                raise KeyError(job_id)
            shutil.rmtree(self.root / job_id)
            del self.records[job_id]

    def _update(self, job_id: str, **updates) -> None:
        with self.lock:
            if job_id not in self.records:
                raise JobCancelled()
            record = {**self.records[job_id], **updates, "updated_at": now()}
            write_json(self.root / job_id / "job.json", record)
            self.records[job_id] = record

    def _work(self) -> None:
        detector = None
        while not self.stopping.is_set():
            job_id = self.queue.get()
            try:
                if job_id is None or self.stopping.is_set():
                    return
                with self.lock:
                    if job_id not in self.records:
                        continue
                    self._update(job_id, status="running", stage="loading", progress=0.0)
                    job = self.get(job_id)
                if detector is None:
                    if self.detector_factory is None:
                        from .detector import Detector

                        detector = Detector(self.settings.weights)
                    else:
                        detector = self.detector_factory(self.settings.weights)
                if self.runner is None:
                    from .pipeline import process_job

                    runner = process_job
                else:
                    runner = self.runner
                capture = json.loads((self.directory(job_id) / "capture.json").read_text())

                def progress(frac: float, stage: str) -> None:
                    if self.stopping.is_set():
                        raise JobCancelled()
                    self._update(job_id, progress=max(0.0, min(1.0, frac)), stage=stage)

                result = runner(job, self.media(job_id), capture, detector, self.settings, progress)
                with self.lock:
                    if job_id not in self.records:
                        continue
                    write_json(self.root / job_id / "result.json", result)
                    self._update(job_id, status="done", stage="done", progress=1.0,
                                 result_url=f"/api/jobs/{job_id}/result")
            except JobCancelled:
                pass
            except Exception as error:
                log.exception("Job %s failed", job_id)
                with self.lock:
                    if job_id in self.records:
                        self._update(job_id, status="failed", stage="failed", error=str(error))
            finally:
                self.queue.task_done()

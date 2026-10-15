"""HTTP lifecycle/auth tests and a real GPU photo/video upload integration."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from fastapi.testclient import TestClient

from cone_app.config import Options, ROOT, Settings
from cone_app.intrinsics import inspect_media, write_json
from cone_app.jobs import JobStore
from cone_app.server import create_app
from cone_app.tests.test_pipeline import make_video, synth_capture

PHOTO = ROOT / "synth_data/images/000001.jpg"


def wait_done(client: TestClient, job_id: str, timeout: float = 90) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "failed":
            raise AssertionError(job["error"])
        if job["status"] == "done":
            return job
        time.sleep(.03)
    raise AssertionError(f"Job {job_id} did not finish")


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "cone_app_data")
        self.settings = Settings(data_dir=Path(self.temporary.name), token=None)
        self.gate = threading.Event()
        self.gate.set()

        def runner(job, media, capture, detector, settings, progress):
            self.gate.wait(5)
            progress(.5, "detecting")
            return {"schema": "cone-result/1", "job_id": job["id"], "frames": []}

        self.store = JobStore(self.settings, runner=runner, detector_factory=lambda _: object())

    def tearDown(self):
        self.gate.set()
        # tempfile.tempdir is app-specific; restore it before deleting that app's disk.
        tempfile.tempdir = str(ROOT / "cone_app_data/tmp")
        self.temporary.cleanup()

    def upload(self, client, endpoint="/api/captures", **data):
        return client.post(endpoint, files={"media": ("frame.jpg", PHOTO.read_bytes(), "image/jpeg")},
                           data={"capture": json.dumps(synth_capture()), **data})

    def test_lifecycle_profiles_range_rerun_delete(self):
        with TestClient(create_app(self.settings, store=self.store)) as client:
            self.assertFalse(client.get("/api/health").json()["auth"])
            self.assertEqual(client.get("/").status_code, 200)
            self.gate.clear()
            response = self.upload(client)
            self.assertEqual(response.status_code, 201, response.text)
            job = response.json()
            self.assertEqual(job["options"], Options().model_dump())
            self.assertEqual(client.get(f"/api/jobs/{job['id']}/result").status_code, 404)
            self.gate.set()
            job = wait_done(client, job["id"])
            self.assertEqual(client.get(job["result_url"]).json()["schema"], "cone-result/1")
            ranged = client.get(job["media_url"], headers={"Range": "bytes=0-31"})
            self.assertEqual(ranged.status_code, 206)
            self.assertEqual(ranged.content, PHOTO.read_bytes()[:32])
            self.assertEqual(ranged.headers["content-range"], f"bytes 0-31/{PHOTO.stat().st_size}")
            self.assertEqual(client.get(job["media_url"], headers={"Range": "bytes=999999999-"}).status_code, 416)
            profiles = client.get("/api/intrinsics-profiles").json()
            self.assertEqual(profiles[0]["n_captures"], 1)
            derived = client.post("/api/jobs", files={"media": ("frame.jpg", PHOTO.read_bytes(), "image/jpeg")},
                                  data={"profile_id": profiles[0]["id"], "name": "profile upload"})
            self.assertEqual(derived.status_code, 201, derived.text)
            wait_done(client, derived.json()["id"])
            rerun = client.post(f"/api/jobs/{job['id']}/rerun", json={"conf": .4})
            self.assertEqual(rerun.status_code, 200, rerun.text)
            self.assertEqual(wait_done(client, job["id"])["options"]["conf"], .4)
            self.assertEqual(len(client.get("/api/jobs").json()), 2)
            self.assertEqual(client.delete(f"/api/jobs/{job['id']}").status_code, 204)
            self.assertEqual(client.get(f"/api/jobs/{job['id']}").status_code, 404)
            self.assertFalse((self.settings.data_dir / "jobs" / job["id"]).exists())

    def test_token_auth_headers_and_media_query(self):
        self.settings.token = "test-secret"
        with TestClient(create_app(self.settings, store=self.store)) as client:
            self.assertTrue(client.get("/api/health").json()["auth"])
            self.assertEqual(client.get("/api/jobs").status_code, 401)
            self.assertEqual(client.get("/api/jobs?token=test-secret").status_code, 401)
            self.assertEqual(self.upload(client).status_code, 401)
            client.headers["Authorization"] = "Bearer test-secret"
            response = self.upload(client)
            self.assertEqual(response.status_code, 201, response.text)
            job = wait_done(client, response.json()["id"])
            del client.headers["Authorization"]
            self.assertEqual(client.get(job["media_url"]).status_code, 401)
            self.assertEqual(client.get(job["media_url"] + "?token=test-secret").status_code, 200)
            self.assertEqual(client.get(job["result_url"] + "?token=test-secret").status_code, 200)

    def test_invalid_capture_and_options(self):
        with TestClient(create_app(self.settings, store=self.store)) as client:
            capture = synth_capture()
            capture["frames"][0]["K"][0][0] = -1
            self.assertEqual(self.upload(client, capture=json.dumps(capture)).status_code, 422)
            self.assertEqual(self.upload(client, options='{"stride": 0}').status_code, 422)
            self.assertEqual(client.get("/api/jobs").json(), [])
            self.assertEqual(list((self.settings.data_dir / "tmp").glob("*")), [])

    def test_restart_requeues_running_and_preserves_completed_jobs(self):
        upload = self.settings.data_dir / "upload.jpg"
        upload.write_bytes(PHOTO.read_bytes())
        record = self.store.create(upload, synth_capture(), inspect_media(PHOTO), Options().model_dump(), "restart", "ios")
        record.update(status="running", stage="detecting", progress=.4)
        write_json(self.store.root / record["id"] / "job.json", record)
        restored = JobStore(self.settings, runner=self.store.runner, detector_factory=lambda _: object())
        self.assertEqual(restored.get(record["id"])["status"], "queued")
        with TestClient(create_app(self.settings, store=restored)) as client:
            wait_done(client, record["id"])
        again = JobStore(self.settings)
        self.assertEqual(again.get(record["id"])["status"], "done")

    def test_failed_job_can_rerun_without_reloading_detector(self):
        loaded = []

        def factory(weights):
            loaded.append(weights)
            return object()

        def runner(job, *args):
            if job["options"]["conf"] == .25:
                raise KeyError("deliberate worker failure")
            return {"schema": "cone-result/1", "job_id": job["id"]}

        self.store.detector_factory = factory
        self.store.runner = runner
        with TestClient(create_app(self.settings, store=self.store)) as client:
            job = self.upload(client).json()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = client.get(f"/api/jobs/{job['id']}").json()
                if current["status"] == "failed":
                    break
                time.sleep(.02)
            self.assertEqual(current["status"], "failed")
            self.assertIn("deliberate worker failure", current["error"])
            response = client.post(f"/api/jobs/{job['id']}/rerun", json={"conf": .4})
            self.assertEqual(response.status_code, 200)
            wait_done(client, job["id"])
            self.assertEqual(len(loaded), 1)

    def test_disk_spooled_upload_preserves_bytes(self):
        payload = PHOTO.read_bytes() + bytes(2 * 1024 * 1024)
        with TestClient(create_app(self.settings, store=self.store)) as client:
            response = client.post("/api/captures", files={"media": ("large.jpg", payload, "image/jpeg")},
                                   data={"capture": json.dumps(synth_capture())})
            self.assertEqual(response.status_code, 201, response.text)
            job = wait_done(client, response.json()["id"])
            self.assertEqual(client.get(job["media_url"]).content, payload)
            self.assertEqual(list((self.settings.data_dir / "tmp").glob("*")), [])


class GPUAPITests(unittest.TestCase):
    def test_real_photo_and_h264_upload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "cone_app_data") as directory:
            settings = Settings(data_dir=Path(directory), token=None)
            video = Path(directory) / "clip.mp4"
            make_video(video, count=6)
            with TestClient(create_app(settings)) as client:
                for path, capture in ((PHOTO, synth_capture()), (video, synth_capture("video", 6))):
                    response = client.post("/api/captures", files={"media": (path.name, path.read_bytes())},
                                           data={"capture": json.dumps(capture)})
                    self.assertEqual(response.status_code, 201, response.text)
                    job = wait_done(client, response.json()["id"])
                    result = client.get(job["result_url"]).json()
                    self.assertGreater(result["stats"]["n_landmarks"], 0)
                    self.assertEqual(result["stats"]["n_frames_processed"], 1 if path == PHOTO else 6)
                    print(f"\nAPI GPU {job['kind']}: {result['stats']}", flush=True)
            tempfile.tempdir = str(ROOT / "cone_app_data/tmp")


if __name__ == "__main__":
    unittest.main()

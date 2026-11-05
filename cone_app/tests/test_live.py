"""Live mode API: ingest, watch relay, logs, auth and recording upload (no GPU or model needed)."""

import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cone_app.config import Settings
from cone_app.jobs import JobStore
from cone_app.live import WATCHER_LIMIT, Watcher
from cone_app.server import create_app

SID = "11111111-2222-3333-4444-555555555555"


def hello(session_id: str = SID, **extra) -> dict:
    return {"type": "hello", "schema": "cone-live/1", "session_id": session_id, "t": 0.0,
            "started_at": "2026-11-05T12:00:00Z", "name": "test run", "device": {"model": "iPhone15,2"},
            "image_size": [1280, 720], **extra}


def frame(i: int, t: float) -> dict:
    return {"type": "frame", "i": i, "t": t, "status": "tracked", "T_world_cam": None, "detections": []}


def map_message(t: float, n: int) -> dict:
    return {"type": "map", "t": t, "keyframes": [],
            "landmarks": [{"id": k, "class_id": 0, "p_world": [k, 0, 0]} for k in range(n)]}


def recording(event: str, t: float, recording_id: str = "rec1") -> dict:
    return {"type": "recording", "t": t, "event": event, "recording_id": recording_id, "name": "lap"}


def jsonl(*messages: dict) -> bytes:
    return "".join(json.dumps(m) + "\n" for m in messages).encode()


def wait_for(check, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError("condition not reached")


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.previous_tempdir = tempfile.tempdir
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = Settings(data_dir=Path(self.temporary.name), token=None)
        self.store = JobStore(self.settings, detector_factory=lambda _: object(),
                              runner=lambda job, *args: {"schema": "cone-result/1", "job_id": job["id"]})

    def tearDown(self):
        tempfile.tempdir = self.previous_tempdir  # create_app points it into the data dir
        self.temporary.cleanup()

    def client(self) -> TestClient:
        return TestClient(create_app(self.settings, store=self.store))

    def log(self, client: TestClient, session_id: str = SID, **params) -> list[dict]:
        response = client.get(f"/api/live/sessions/{session_id}/log", params=params)
        return [json.loads(line) for line in response.text.splitlines()]

    def test_ingest_list_get_log(self):
        with self.client() as client:
            self.assertEqual(client.get("/api/live/sessions").json(), [])
            with client.websocket_connect("/api/live/ingest") as ws:
                for message in (hello(), map_message(.5, 3), frame(0, .1), frame(1, .2), recording("start", .3),
                                recording("stop", .4), {"type": "bye", "t": .5}):
                    ws.send_json(message)
                wait_for(lambda: client.get(f"/api/live/sessions/{SID}").json()["duration_s"] == .5)
                self.assertTrue(client.get(f"/api/live/sessions/{SID}").json()["active"])
            wait_for(lambda: not client.get(f"/api/live/sessions/{SID}").json()["active"])
            session = client.get(f"/api/live/sessions/{SID}").json()
            self.assertEqual((session["name"], session["device"]["model"], session["n_frames"],
                              session["n_landmarks"]), ("test run", "iPhone15,2", 2, 3))
            self.assertEqual(session["recordings"], [{"recording_id": "rec1", "name": "lap", "t_start": .3,
                                                      "t_end": .4, "job_id": None, "has_log": False}])
            self.assertEqual([s["id"] for s in client.get("/api/live/sessions").json()], [SID])
            response = client.get(f"/api/live/sessions/{SID}/log")
            self.assertTrue(response.headers["content-type"].startswith("application/x-ndjson"))
            self.assertEqual([m["type"] for m in self.log(client)],
                             ["hello", "map", "frame", "frame", "recording", "recording", "bye"])
            self.assertEqual(client.get("/api/live/sessions/nope/log").status_code, 404)
            self.assertEqual(client.get("/api/live/sessions/..%2f..%2fjobs").status_code, 404)
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log?recording_id=..%2fsession").status_code, 404)
            # Metadata survives a restart.
        with self.client() as client:
            self.assertEqual(client.get(f"/api/live/sessions/{SID}").json()["n_frames"], 2)

    def test_first_message_must_be_hello(self):
        with self.client() as client:
            with client.websocket_connect("/api/live/ingest") as ws:
                ws.send_json(frame(0, 0))
                with self.assertRaises(WebSocketDisconnect) as caught:
                    ws.receive_text()
            self.assertEqual(caught.exception.code, 4400)
            self.assertEqual(client.get("/api/live/sessions").json(), [])

    def test_watcher_gets_latest_state_then_relay(self):
        with self.client() as client:
            with client.websocket_connect("/api/live/ingest") as phone:
                for message in (hello(), map_message(.1, 2), frame(0, .1), frame(1, .2)):
                    phone.send_json(message)
                wait_for(lambda: client.get(f"/api/live/sessions/{SID}").json()["n_frames"] == 2)
                with client.websocket_connect(f"/api/live/sessions/{SID}/watch") as watcher:
                    first = [watcher.receive_json() for _ in range(3)]
                    self.assertEqual([m["type"] for m in first], ["hello", "map", "frame"])
                    self.assertEqual(first[2]["i"], 1)
                    phone.send_json(frame(2, .3))
                    self.assertEqual(watcher.receive_json()["i"], 2)
                    phone.send_json(recording("start", .35))
                    self.assertEqual(watcher.receive_json()["type"], "recording")
                # Leave without `bye`.
            with client.websocket_connect(f"/api/live/sessions/{SID}/watch") as watcher:
                # A late watcher gets the cached state of the now inactive session.
                self.assertEqual([watcher.receive_json()["type"] for _ in range(3)], ["hello", "map", "frame"])
            self.assertFalse(client.get(f"/api/live/sessions/{SID}").json()["active"])

    def test_live_watcher_sees_disconnected_and_bye_closes(self):
        with self.client() as client:
            with client.websocket_connect("/api/live/ingest") as phone:
                phone.send_json(hello())
                wait_for(lambda: client.get(f"/api/live/sessions/{SID}").json()["active"])
                with client.websocket_connect(f"/api/live/sessions/{SID}/watch") as watcher:
                    self.assertEqual(watcher.receive_json()["type"], "hello")
                    phone.close()
                    self.assertEqual(watcher.receive_json()["type"], "disconnected")
            with client.websocket_connect("/api/live/ingest") as phone:
                phone.send_json(hello())
                with client.websocket_connect(f"/api/live/sessions/{SID}/watch") as watcher:
                    watcher.receive_json()
                    phone.send_json({"type": "bye", "t": 1.0})
                    self.assertEqual(watcher.receive_json()["type"], "bye")
                    with self.assertRaises(WebSocketDisconnect):
                        watcher.receive_json()

    def test_watch_unknown_session_closes_4404(self):
        with self.client() as client:
            with client.websocket_connect("/api/live/sessions/missing/watch") as watcher:
                with self.assertRaises(WebSocketDisconnect) as caught:
                    watcher.receive_text()
            self.assertEqual(caught.exception.code, 4404)

    def test_reconnect_appends(self):
        with self.client() as client:
            with client.websocket_connect("/api/live/ingest") as ws:
                for message in (hello(), frame(0, .1)):
                    ws.send_json(message)
                wait_for(lambda: client.get(f"/api/live/sessions/{SID}").json()["n_frames"] == 1)
            with client.websocket_connect("/api/live/ingest") as ws:
                for message in (hello(), frame(1, 5.0), frame(2, 5.1)):
                    ws.send_json(message)
                wait_for(lambda: client.get(f"/api/live/sessions/{SID}").json()["n_frames"] == 3)
            self.assertEqual(len(client.get("/api/live/sessions").json()), 1)
            self.assertEqual([m["type"] for m in self.log(client)], ["hello", "frame", "hello", "frame", "frame"])
            self.assertEqual(client.get(f"/api/live/sessions/{SID}").json()["duration_s"], 5.1)

    def test_token_auth_for_websockets_and_log(self):
        self.settings.token = "secret"
        with self.client() as client:
            for path in ("/api/live/ingest", "/api/live/ingest?token=wrong", f"/api/live/sessions/{SID}/watch"):
                with client.websocket_connect(path) as ws:
                    with self.assertRaises(WebSocketDisconnect) as caught:
                        ws.receive_text()
                self.assertEqual(caught.exception.code, 4401, path)
            with client.websocket_connect("/api/live/ingest?token=secret") as ws:
                ws.send_json(hello())
                wait_for(lambda: client.get(f"/api/live/sessions/{SID}", headers={"Authorization": "Bearer secret"}).json())
            with client.websocket_connect("/api/live/ingest", headers={"Authorization": "Bearer secret"}) as ws:
                ws.send_json(frame(0, 0))  # not a hello: closed with 4400, proving auth passed
                with self.assertRaises(WebSocketDisconnect) as caught:
                    ws.receive_text()
                self.assertEqual(caught.exception.code, 4400)
            with client.websocket_connect(f"/api/live/sessions/{SID}/watch?token=secret") as watcher:
                self.assertEqual(watcher.receive_json()["type"], "hello")
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log").status_code, 401)
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log?token=wrong").status_code, 401)
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log?token=secret").status_code, 200)
            self.assertEqual(client.get(f"/api/live/sessions/{SID}?token=secret").status_code, 401)

    def test_recording_slice_of_stream(self):
        with self.client() as client:
            messages = [hello(), map_message(.1, 1), frame(0, .1), map_message(.5, 2), frame(1, .6),
                        recording("start", .7), frame(2, .8), frame(3, .9), recording("stop", 1.0), frame(4, 1.1)]
            self.stream_all(client, messages)
            sliced = self.log(client, recording_id="rec1")
            self.assertEqual(sliced, [messages[i] for i in (0, 3, 5, 6, 7, 8)])
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log?recording_id=other").status_code, 404)

    def stream_all(self, client: TestClient, messages: list[dict]):
        with client.websocket_connect("/api/live/ingest") as ws:
            for message in messages:
                ws.send_json(message)
            wait_for(lambda: len(self.log(client)) == len(messages))

    def test_slow_watcher_drops_frames_not_state(self):
        watcher = Watcher()
        watcher.push("hello", "h")
        for i in range(WATCHER_LIMIT * 2):
            watcher.push("frame", str(i))
        watcher.push("bye", "b")
        kinds = [kind for kind, _ in watcher.items]
        self.assertEqual((kinds[0], kinds[-1]), ("hello", "bye"))
        self.assertLessEqual(len(kinds), WATCHER_LIMIT + 1)

    def capture_form(self, **extra) -> dict:
        capture = {"schema": "cone-capture/1", "kind": "photo", "device": {"model": "test"},
                   "created_at": "2026-11-05T12:00:00Z", "image_size": [64, 48], "distortion": None,
                   "frames": [{"index": 0, "t": 0, "K": [[50, 0, 32], [0, 50, 24], [0, 0, 1]]}]}
        return {"capture": json.dumps(capture), **extra}

    def photo(self) -> tuple[str, bytes, str]:
        return "frame.jpg", cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes(), "image/jpeg"

    def test_capture_with_live_log_links_job(self):
        log = jsonl(hello(), map_message(.1, 2), frame(0, .2), recording("start", .1))
        with self.client() as client:
            # Never streamed: the session is created from the log's hello.
            response = client.post("/api/captures", files={"media": self.photo(), "live_log": ("log.jsonl", log)},
                                   data=self.capture_form(live_session_id=SID, recording_id="rec1"))
            self.assertEqual(response.status_code, 201, response.text)
            job_id = response.json()["id"]
            session = client.get(f"/api/live/sessions/{SID}").json()
            self.assertEqual((session["name"], session["active"]), ("test run", False))
            self.assertEqual(session["recordings"][0]["job_id"], job_id)
            self.assertTrue(session["recordings"][0]["has_log"])
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log?recording_id=rec1").content, log)
            self.assertEqual(client.get(f"/api/live/sessions/{SID}/log").content, b"")
            self.assertEqual(list((self.settings.data_dir / "tmp").glob("*")), [])
            # Streamed session: the job id is linked to the recording from the stream.
            self.stream_all(client, [hello(), recording("start", 1.0, "rec2"), recording("stop", 2.0, "rec2")])
            other = client.post("/api/captures", files={"media": self.photo()},
                                data=self.capture_form(live_session_id=SID, recording_id="rec2"))
            self.assertEqual(other.status_code, 201, other.text)
            entries = {r["recording_id"]: r for r in client.get(f"/api/live/sessions/{SID}").json()["recordings"]}
            self.assertEqual((entries["rec2"]["job_id"], entries["rec2"]["has_log"]), (other.json()["id"], False))
            self.assertEqual(entries["rec2"]["t_end"], 2.0)

    def test_capture_rejects_bad_live_fields_and_plain_capture_unchanged(self):
        with self.client() as client:
            bad = client.post("/api/captures", files={"media": self.photo(), "live_log": ("log.jsonl", jsonl(frame(0, 0)))},
                              data=self.capture_form(live_session_id=SID, recording_id="rec1"))
            self.assertEqual(bad.status_code, 422)
            traversal = client.post("/api/captures", files={"media": self.photo(), "live_log": ("log.jsonl", jsonl(hello()))},
                                    data=self.capture_form(live_session_id=SID, recording_id="../evil"))
            self.assertEqual(traversal.status_code, 422)
            self.assertEqual(client.get("/api/jobs").json(), [])
            self.assertEqual(client.get("/api/live/sessions").json(), [])
            plain = client.post("/api/captures", files={"media": self.photo()}, data=self.capture_form())
            self.assertEqual(plain.status_code, 201, plain.text)
            self.assertEqual(client.get("/api/live/sessions").json(), [])

    def test_delete(self):
        with self.client() as client:
            self.stream_all(client, [hello(), frame(0, .1)])
            self.assertEqual(client.delete(f"/api/live/sessions/{SID}").status_code, 204)
            self.assertEqual(client.get(f"/api/live/sessions/{SID}").status_code, 404)
            self.assertEqual(client.delete(f"/api/live/sessions/{SID}").status_code, 404)
            self.assertFalse((self.settings.data_dir / "live" / SID).exists())


if __name__ == "__main__":
    unittest.main()

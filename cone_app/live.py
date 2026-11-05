"""Live mode (schema ``cone-live/1``): session storage, watcher relay and the ``/api/live`` routes."""

import asyncio
from collections import deque
import hmac
import json
from pathlib import Path
import re
import shutil
import threading
from typing import Iterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse

from .config import Settings
from .intrinsics import now, write_json

SCHEMA = "cone-live/1"
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
ACK_INTERVAL_S = 2.0
WATCHER_LIMIT = 256  # queued messages before a slow watcher starts losing frames
NDJSON = "application/x-ndjson"


def valid_id(value: str) -> bool:
    return ID_PATTERN.fullmatch(value) is not None


def encode(message: dict) -> str:
    return json.dumps(message, separators=(",", ":"))


class Watcher:
    """Queue for one browser. Under backpressure the oldest ``frame`` is dropped, nothing else."""

    def __init__(self) -> None:
        self.items: deque[tuple[str, str]] = deque()
        self.ready = asyncio.Event()

    def push(self, kind: str, text: str) -> None:
        if len(self.items) >= WATCHER_LIMIT:
            for index, (queued, _) in enumerate(self.items):
                if queued == "frame":
                    del self.items[index]
                    break
        self.items.append((kind, text))
        self.ready.set()

    async def get(self) -> tuple[str, str]:
        while not self.items:
            self.ready.clear()
            await self.ready.wait()
        return self.items.popleft()


class Session:
    """In-memory state of one session: persisted metadata plus the latest hello/map/frame."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta
        self.latest: dict[str, tuple[str, str]] = {}
        self.watchers: set[Watcher] = set()
        self.connections = 0
        self.dirty = False
        self.deleted = False


class LiveStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sessions: dict[str, Session] = {}

    # --- paths -------------------------------------------------------------------------------
    def directory(self, session_id: str) -> Path:
        return self.root / session_id

    def stream_path(self, session_id: str) -> Path:
        return self.directory(session_id) / "stream.jsonl"

    def recording_path(self, session_id: str, recording_id: str) -> Path:
        return self.directory(session_id) / "recordings" / f"{recording_id}.jsonl"

    # --- sessions ----------------------------------------------------------------------------
    def _get(self, session_id: str) -> Session | None:
        """Load a session from memory, else from disk (rebuilding the latest-message cache)."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                try:
                    meta = json.loads((self.directory(session_id) / "session.json").read_text())
                except (OSError, ValueError):
                    return None
                session = self.sessions[session_id] = Session(meta)
                for line in self._lines(self.stream_path(session_id)):
                    try:
                        kind = json.loads(line).get("type")
                    except (ValueError, AttributeError):
                        continue
                    if kind in ("hello", "map", "frame"):
                        session.latest[kind] = (kind, line)
            return session

    @staticmethod
    def _lines(path: Path) -> Iterator[str]:
        try:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    if line.endswith("\n") and line.strip():
                        yield line.rstrip("\n")
        except OSError:
            return

    def _save(self, session_id: str, session: Session) -> None:
        if not session.deleted:
            write_json(self.directory(session_id) / "session.json", session.meta)
            session.dirty = False

    def _new_meta(self, session_id: str, hello: dict) -> dict:
        stamp = now()
        return {"id": session_id, "name": "", "device": {}, "started_at": hello.get("started_at") or stamp,
                "updated_at": stamp, "duration_s": 0.0, "n_frames": 0, "n_landmarks": 0, "recordings": []}

    def _ensure(self, session_id: str, hello: dict) -> Session:
        with self.lock:
            session = self._get(session_id)
            if session is None:
                self.directory(session_id).mkdir(parents=True, exist_ok=True)
                session = self.sessions[session_id] = Session(self._new_meta(session_id, hello))
                self._save(session_id, session)
            return session

    def view(self, session_id: str) -> dict:
        with self.lock:
            session = self._get(session_id)
            if session is None:
                raise KeyError(session_id)
            recordings = [{**r, "has_log": self.recording_path(session_id, r["recording_id"]).is_file()}
                          for r in session.meta["recordings"]]
            return {**session.meta, "recordings": recordings, "active": session.connections > 0}

    def list(self) -> list[dict]:
        ids = [path.name for path in self.root.iterdir() if path.is_dir() and valid_id(path.name)]
        views = []
        for session_id in ids:
            try:
                views.append(self.view(session_id))
            except KeyError:
                continue
        return sorted(views, key=lambda view: view["started_at"], reverse=True)

    def delete(self, session_id: str) -> None:
        with self.lock:
            session = self._get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.deleted = True
            self.sessions.pop(session_id, None)
            shutil.rmtree(self.directory(session_id), ignore_errors=True)
            self._publish(session, {"type": "bye", "t": session.meta["duration_s"]})

    # --- ingest ------------------------------------------------------------------------------
    def open_ingest(self, hello: dict) -> tuple[str, Session]:
        session_id = hello["session_id"]
        with self.lock:
            session = self._ensure(session_id, hello)
            session.connections += 1
        return session_id, session

    def close_ingest(self, session_id: str, session: Session, said_bye: bool) -> None:
        with self.lock:
            session.connections -= 1
            if not session.deleted:
                self._save(session_id, session)
                if session.connections == 0 and not said_bye:
                    self._publish(session, {"type": "disconnected", "t": session.meta["duration_s"]})

    def ingest(self, session_id: str, session: Session, message: dict) -> bool:
        """Append one message, update metadata and relay it. False if the session was deleted."""
        text = encode(message)
        with self.lock:
            if session.deleted:
                return False
            with self.stream_path(session_id).open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")
            meta, kind = session.meta, message["type"]
            meta["updated_at"] = now()
            if isinstance(message.get("t"), (int, float)):
                meta["duration_s"] = float(message["t"])
            if kind == "hello":
                meta["name"] = str(message.get("name") or meta["name"])
                meta["device"] = message.get("device") if isinstance(message.get("device"), dict) else meta["device"]
            elif kind == "frame":
                meta["n_frames"] += 1
            elif kind == "map" and isinstance(message.get("landmarks"), list):
                meta["n_landmarks"] = len(message["landmarks"])
            elif kind == "recording" and valid_id(str(message.get("recording_id", ""))):
                self._recording_event(meta, message)
            if kind in ("hello", "map", "frame"):
                session.latest[kind] = (kind, text)
            session.dirty = True
            if kind not in ("frame", "map"):
                self._save(session_id, session)
            self._publish(session, message, text)
            return True

    @staticmethod
    def _recording_entry(meta: dict, recording_id: str) -> dict:
        for entry in meta["recordings"]:
            if entry["recording_id"] == recording_id:
                return entry
        entry = {"recording_id": recording_id, "name": "", "t_start": None, "t_end": None, "job_id": None}
        meta["recordings"].append(entry)
        return entry

    def _recording_event(self, meta: dict, message: dict) -> None:
        entry = self._recording_entry(meta, message["recording_id"])
        if message.get("event") == "start":
            entry.update(name=str(message.get("name") or entry["name"]), t_start=message.get("t"), t_end=None)
        elif message.get("event") == "stop":
            entry["t_end"] = message.get("t")

    def flush(self, session_id: str, session: Session) -> None:
        with self.lock:
            if session.dirty:
                self._save(session_id, session)

    def _publish(self, session: Session, message: dict, text: str | None = None) -> None:
        text = text or encode(message)
        for watcher in session.watchers:
            watcher.push(message["type"], text)

    # --- watch -------------------------------------------------------------------------------
    def subscribe(self, session_id: str) -> tuple[Session, Watcher]:
        """Register a watcher primed with the latest state; nothing can be missed in between."""
        with self.lock:
            session = self._get(session_id)
            if session is None:
                raise KeyError(session_id)
            watcher = Watcher()
            for kind in ("hello", "map", "frame"):
                if kind in session.latest:
                    watcher.push(*session.latest[kind])
            session.watchers.add(watcher)
            return session, watcher

    # --- logs --------------------------------------------------------------------------------
    def log_lines(self, session_id: str, recording_id: str | None) -> Iterator[str]:
        """Full stream, or one recording sliced out of it (with the preceding hello and map)."""
        if recording_id is None:
            yield from self._lines(self.stream_path(session_id))
            return
        hello = map_ = None
        inside = False
        for line in self._lines(self.stream_path(session_id)):
            message = json.loads(line)
            kind = message.get("type")
            is_mine = kind == "recording" and message.get("recording_id") == recording_id
            if not inside:
                if kind == "hello":
                    hello = line
                elif kind == "map":
                    map_ = line
                elif is_mine and message.get("event") == "start":
                    inside = True
                    yield from (part for part in (hello, map_) if part)
                    yield line
                continue
            yield line
            if is_mine and message.get("event") == "stop":
                return

    # --- capture upload ----------------------------------------------------------------------
    def check_log(self, session_id: str, log: Path) -> None:
        """Raise ValueError unless an uploaded log can be stored for this session."""
        if self._get(session_id) is not None:
            return
        first = next(self._lines(log), None)
        try:
            hello = json.loads(first or "")
        except ValueError:
            hello = None
        if not (isinstance(hello, dict) and hello.get("type") == "hello" and hello.get("schema") == SCHEMA
                and hello.get("session_id") == session_id):
            raise ValueError(f"live_log must start with a {SCHEMA} hello for session {session_id}")

    def link_recording(self, session_id: str, recording_id: str, job_id: str, log: Path | None) -> None:
        with self.lock:
            session = self._get(session_id)
            if log is not None:
                if session is None:
                    hello = json.loads(next(self._lines(log)))
                    session = self._ensure(session_id, hello)
                    session.meta.update(name=str(hello.get("name") or ""),
                                        device=hello.get("device") if isinstance(hello.get("device"), dict) else {})
                target = self.recording_path(session_id, recording_id)
                target.parent.mkdir(exist_ok=True)
                log.replace(target)
            if session is not None:
                self._recording_entry(session.meta, recording_id)["job_id"] = job_id
                session.meta["updated_at"] = now()
                self._save(session_id, session)


def register(app: FastAPI, settings: Settings) -> LiveStore:
    """Mount the ``/api/live`` routes on ``app`` and return their store."""
    store = LiveStore(settings.data_dir / "live")

    async def authorized(websocket: WebSocket) -> bool:
        """Accept, then close with 4401 on a bad token so browsers can see the code."""
        await websocket.accept()
        if settings.token:
            header = websocket.headers.get("authorization", "")
            supplied = header[7:] if header.startswith("Bearer ") else websocket.query_params.get("token", "")
            if not hmac.compare_digest(supplied.encode(), settings.token.encode()):
                await websocket.close(code=4401)
                return False
        return True

    def session_id_or_404(session_id: str) -> str:
        if not valid_id(session_id):
            raise HTTPException(status_code=404, detail="Live session not found")
        return session_id

    @app.websocket("/api/live/ingest")
    async def ingest(websocket: WebSocket):
        if not await authorized(websocket):
            return
        try:
            hello = json.loads(await websocket.receive_text())
        except (ValueError, KeyError):
            hello = None
        except WebSocketDisconnect:
            return
        if not (isinstance(hello, dict) and hello.get("type") == "hello" and hello.get("schema") == SCHEMA
                and isinstance(hello.get("session_id"), str) and valid_id(hello["session_id"])):
            await websocket.close(code=4400, reason=f"first message must be a {SCHEMA} hello")
            return
        session_id, session = store.open_ingest(hello)

        async def acks() -> None:
            try:
                while True:
                    await asyncio.sleep(ACK_INTERVAL_S)
                    store.flush(session_id, session)
                    await websocket.send_text(encode({"type": "ack", "t": session.meta["duration_s"]}))
            except Exception:  # the receive loop notices the closed socket
                return

        ack_task = asyncio.create_task(acks())
        said_bye = False
        try:
            message = hello
            while True:
                if isinstance(message, dict) and isinstance(message.get("type"), str):
                    if message.get("type") == "hello" and message.get("session_id") != session_id:
                        await websocket.close(code=4400, reason="session_id changed")
                        return
                    if not store.ingest(session_id, session, message):
                        await websocket.close(code=4404, reason="session deleted")
                        return
                    said_bye = said_bye or message["type"] == "bye"
                try:
                    message = json.loads(await websocket.receive_text())
                except ValueError:
                    message = None  # ignore malformed messages
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            ack_task.cancel()
            store.close_ingest(session_id, session, said_bye)

    @app.get("/api/live/sessions")
    def list_sessions():
        return store.list()

    @app.get("/api/live/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            return store.view(session_id_or_404(session_id))
        except KeyError:
            raise HTTPException(status_code=404, detail="Live session not found") from None

    @app.get("/api/live/sessions/{session_id}/log")
    def get_log(session_id: str, recording_id: str | None = None):
        view = get_session(session_id)
        if recording_id is None:
            path = store.stream_path(session_id)
            return FileResponse(path, media_type=NDJSON) if path.is_file() else Response(media_type=NDJSON)
        if not valid_id(recording_id):
            raise HTTPException(status_code=404, detail="Recording not found")
        path = store.recording_path(session_id, recording_id)
        if path.is_file():
            return FileResponse(path, media_type=NDJSON)
        if not any(r["recording_id"] == recording_id and r["t_start"] is not None for r in view["recordings"]):
            raise HTTPException(status_code=404, detail="Recording not found")
        return StreamingResponse((line + "\n" for line in store.log_lines(session_id, recording_id)),
                                 media_type=NDJSON)

    @app.websocket("/api/live/sessions/{session_id}/watch")
    async def watch(websocket: WebSocket, session_id: str):
        if not await authorized(websocket):
            return
        try:
            session, watcher = store.subscribe(session_id_or_404(session_id))
        except (KeyError, HTTPException):
            await websocket.close(code=4404, reason="Live session not found")
            return

        async def relay() -> None:
            try:
                while True:
                    kind, text = await watcher.get()
                    await websocket.send_text(text)
                    if kind == "bye":
                        await websocket.close()
                        return
            except Exception:  # the browser went away
                return

        async def until_disconnect() -> None:
            try:
                while True:
                    await websocket.receive_text()
            except Exception:  # WebSocketDisconnect when the browser leaves
                return

        tasks = [asyncio.create_task(relay()), asyncio.create_task(until_disconnect())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            session.watchers.discard(watcher)

    @app.delete("/api/live/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        try:
            store.delete(session_id_or_404(session_id))  # on the loop: it notifies watchers
        except KeyError:
            raise HTTPException(status_code=404, detail="Live session not found") from None
        return Response(status_code=204)

    return store

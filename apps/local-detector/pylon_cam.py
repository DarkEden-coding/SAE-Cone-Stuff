"""Basler pylon cameras (GigE / USB3 Vision), which browsers can't access directly."""
import threading

try:
    from pypylon import pylon
except ImportError:  # pylon support is optional
    pylon = None

_in_use: set[str] = set()
_in_use_lock = threading.Lock()


def list_cameras() -> list[dict]:
    if pylon is None:
        return []
    return [{"id": d.GetSerialNumber(), "label": f"Basler {d.GetModelName()} ({d.GetSerialNumber()})"}
            for d in pylon.TlFactory.GetInstance().EnumerateDevices()]


class PylonCamera:
    """Opens a camera by serial number and continuously acquires, keeping only the newest frame."""

    def __init__(self, serial: str):
        if pylon is None:
            raise RuntimeError("pypylon is not installed")
        with _in_use_lock:
            if serial in _in_use:
                raise RuntimeError("camera is already streaming in another tab")
            _in_use.add(serial)
        self.serial, self.cam = serial, None
        self.lock = threading.Lock()  # grab() and close() may run on different threads
        try:
            factory = pylon.TlFactory.GetInstance()
            info = next((d for d in factory.EnumerateDevices() if d.GetSerialNumber() == serial), None)
            if info is None:
                raise RuntimeError(f"camera {serial} not found")
            self.cam = pylon.InstantCamera(factory.CreateDevice(info))
            try:
                self.cam.Open()
            except Exception as e:
                if "controlled by another application" in str(e):
                    raise RuntimeError("camera is in use by another application (e.g. pylon Viewer)") from e
                raise
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            self._close()
            raise

    def grab(self, timeout_ms: int = 2000, attempts: int = 5):
        """Return the newest frame as a BGR array. Retries incomplete frames (common on busy GigE links)."""
        with self.lock:
            return self._grab(timeout_ms, attempts)

    def _grab(self, timeout_ms: int, attempts: int):
        error = "no frame"
        for _ in range(attempts):
            result = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            try:
                if result.GrabSucceeded():
                    return self.converter.Convert(result).GetArray()
                error = result.ErrorDescription
            finally:
                result.Release()
        raise RuntimeError(f"grab failed: {error}")

    def close(self):
        with self.lock:
            self._close()

    def _close(self):
        try:
            if self.cam is not None:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
                if self.cam.IsOpen():
                    self.cam.Close()
        finally:
            with _in_use_lock:
                _in_use.discard(self.serial)

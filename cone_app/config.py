"""Application paths and validated processing options."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs/by-n-E-far/weights/best.pt"
CORRECTION_TABLES = {
    (str(DEFAULT_WEIGHTS.resolve()), 1920): ROOT / "cone_app/assets/kp_correction_E_1920.json",
}


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    stride: int | None = Field(default=None, ge=1)
    target_fps: float = Field(default=10, gt=0, le=240)
    imgsz: int = Field(default=1920, ge=32, le=4096)
    conf: float = Field(default=0.25, ge=0, le=1)
    kp_conf: float = Field(default=0.5, ge=0, le=1)
    kp_correction: bool = True
    flat_ground: bool = True
    max_range_m: float = Field(default=40.0, gt=0)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CONE_APP_DATA_DIR", ROOT / "cone_app_data")))
    weights: Path = field(default_factory=lambda: Path(os.environ.get("CONE_APP_WEIGHTS", DEFAULT_WEIGHTS)))
    imgsz: int = field(default_factory=lambda: int(os.environ.get("CONE_APP_IMGSZ", "1920")))
    batch_size: int = field(default_factory=lambda: max(1, int(os.environ.get("CONE_APP_BATCH_SIZE", "2"))))
    token: str | None = field(default_factory=lambda: os.environ.get("CONE_APP_TOKEN") or None)
    static_dir: Path = ROOT / "cone_app/static"

    def correction_path(self, imgsz: int) -> Path | None:
        path = CORRECTION_TABLES.get((str(self.weights.resolve()), imgsz))
        return path if path and path.is_file() else None

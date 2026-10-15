"""Upload images or videos for blue/yellow cone detection. Run with uv run web_view.py."""

import argparse
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs/yolo26m-640-small-cones/weights/best.pt"


def load_model(weights: Path) -> YOLO:
    """Load the requested local checkpoint and reject single-class models."""
    if not weights.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights}")
    model = YOLO(str(weights))
    if len(model.names) < 2:
        raise ValueError(f"Expected a multiclass checkpoint, found {model.names}")
    return model


def build_app(model: YOLO, weights: Path) -> gr.Blocks:
    """Build upload, inference, and playback controls around one resident model."""
    device = 0 if torch.cuda.is_available() else "cpu"

    def predict_image(
        image: Image.Image | None, confidence: float
    ) -> tuple[np.ndarray, str]:
        """Annotate an uploaded image and report counts by class."""
        if image is None:
            raise gr.Error("Upload an image first.")
        started = perf_counter()
        result = model.predict(
            image, conf=confidence, imgsz=640, device=device, verbose=False
        )[0]
        counts = Counter(model.names[int(label)] for label in result.boxes.cls.tolist())
        summary = ", ".join(f"{name}: {counts[name]}" for name in model.names.values())
        return result.plot()[:, :, ::-1], f"{summary} | {perf_counter() - started:.2f}s"

    def predict_video(
        video: str | None,
        confidence: float,
        progress: gr.Progress = gr.Progress(),  # noqa: B008 Gradio injects progress via this default.
    ) -> tuple[str, str]:
        """Annotate every decoded frame and encode an MP4 with optional source audio."""
        if not video:
            raise gr.Error("Upload a video first.")
        if not shutil.which("ffmpeg"):
            raise gr.Error(
                "Install ffmpeg with the libopenh264 encoder for video playback."
            )
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            capture.release()
            raise gr.Error("Cannot decode this video. Try an MP4, MOV, or WebM file.")
        started = perf_counter()
        writer = None
        frames = 0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise gr.Error(
                "The video has no valid frame rate. Re-encode it before uploading."
            )
        try:
            with tempfile.TemporaryDirectory(prefix="sae-video-") as directory:
                raw = Path(directory) / "annotated.avi"
                output = Path(directory) / "annotated.mp4"
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(raw),
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise gr.Error(
                                "Cannot create the annotated video. Check disk space and OpenCV codecs."
                            )
                    result = model.predict(
                        frame, conf=confidence, imgsz=640, device=device, verbose=False
                    )[0]
                    writer.write(result.plot())
                    frames += 1
                    if frames % 10 == 0:
                        progress(
                            (frames, max(frames, total)),
                            desc=f"Detecting frame {frames}",
                        )
                if frames == 0:
                    raise gr.Error("The video contains no readable frames.")
                if total > 0 and frames < total:
                    raise gr.Error(
                        f"Video decoding stopped at frame {frames} of {total}. Try re-encoding the input."
                    )
                writer.release()
                progress(0.95, desc="Encoding browser-playable MP4")
                encoded = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(raw),
                        "-i",
                        video,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0?",
                        "-c:v",
                        "libopenh264",
                        "-b:v",
                        "5M",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        "-shortest",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if encoded.returncode:
                    raise gr.Error(f"MP4 encoding failed: {encoded.stderr[-1500:]}")
                # Gradio owns this cached copy; the large intermediate AVI is deleted immediately.
                cached = gr.processing_utils.save_file_to_cache(
                    str(output), gr.utils.get_upload_folder()
                )
                return (
                    cached,
                    f"{frames} frames at {fps:.2f} FPS | {perf_counter() - started:.2f}s",
                )
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    with gr.Blocks(
        title="SAE cone detection", analytics_enabled=False, delete_cache=(3600, 86400)
    ) as app:
        gr.Markdown(
            f"# SAE cone detection\nModel: `{weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}`\n\n"
            f"Classes: {', '.join(model.names.values())}. Device: {'CUDA GPU' if device == 0 else 'CPU'}. "
            "Upload an image or video, then run detection. Videos are processed completely before playback. "
            "Uploads and results expire after 24 hours. Maximum upload: 500 MB."
        )
        confidence = gr.Slider(
            0.01, 0.95, value=0.25, step=0.01, label="Confidence threshold"
        )
        # ponytail: one shared GPU queue; add workers only if concurrent throughput matters.
        with gr.Tab("Image"):
            with gr.Row():
                image = gr.Image(type="pil", sources=["upload"], label="Input image")
                annotated_image = gr.Image(label="Detections", interactive=False)
            image_status = gr.Textbox(label="Detection counts", interactive=False)
            gr.Button("Detect image", variant="primary").click(
                predict_image,
                [image, confidence],
                [annotated_image, image_status],
                api_name="detect_image",
                concurrency_id="inference",
                concurrency_limit=1,
            )
        with gr.Tab("Video"):
            with gr.Row():
                video = gr.Video(sources=["upload"], label="Input video")
                annotated_video = gr.Video(
                    label="Annotated playback", interactive=False
                )
            video_status = gr.Textbox(label="Processing summary", interactive=False)
            gr.Button("Detect video", variant="primary").click(
                predict_video,
                [video, confidence],
                [annotated_video, video_status],
                api_name="detect_video",
                concurrency_id="inference",
                concurrency_limit=1,
            )
    return app.queue(max_size=8, default_concurrency_limit=1)


def main() -> None:
    """Launch a loopback-only server for local use or a private Tailscale proxy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=WEIGHTS)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    weights = args.model.resolve()
    app = build_app(load_model(weights), weights)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        max_file_size="500mb",
        blocked_paths=[str(ROOT)],
    )


if __name__ == "__main__":
    main()

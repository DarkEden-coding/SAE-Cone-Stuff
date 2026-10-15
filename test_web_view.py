"""Real image/video regression check against a running app: uv run test_web_view.py."""

import tempfile
from pathlib import Path

import cv2
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Upload a dataset image and short clip, then verify annotated downloads."""
    client = Client("http://127.0.0.1:7860", verbose=False)
    image = next((ROOT / "dataset/valid/images").glob("*.jpg"))
    result, summary = client.predict(
        handle_file(str(image)), 0.25, api_name="/detect_image"
    )
    assert Path(result).is_file(), result
    assert "blue_cone:" in summary and "yellow_cone:" in summary, summary
    with tempfile.TemporaryDirectory() as directory:
        video = Path(directory) / "input.mp4"
        frame = cv2.imread(str(image))
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
        )
        assert writer.isOpened()
        for _ in range(5):
            writer.write(frame)
        writer.release()
        result, summary = client.predict(
            handle_file(str(video)), 0.25, api_name="/detect_video"
        )
        capture = cv2.VideoCapture(result)
        try:
            assert capture.isOpened(), result
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5, summary
            assert abs(capture.get(cv2.CAP_PROP_FPS) - 10) < 0.1
            assert capture.read()[0]
        finally:
            capture.release()
        assert "5 frames" in summary, summary
    try:
        client.predict(None, 0.25, api_name="/detect_image")
    except AppError as error:
        assert "Upload an image first" in str(error), error
    else:
        raise AssertionError("Empty image upload was accepted")
    print(
        "Image inference, video inference/playback, and empty-input validation passed."
    )


if __name__ == "__main__":
    main()

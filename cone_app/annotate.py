"""Render a job's media with its detection overlay burned in (boxes, keypoints, labels).

Sizes follow each cone's box height, like the web overlay, so distant cones get small marks.
Video frames between processed frames reuse the most recent processed frame's detections.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# BGR versions of static/util.js CLASS_COLORS
_COLORS = {'BLUE': (255, 111, 47), 'YELLOW': (0, 212, 255), 'ORANGE': (0, 122, 255),
           'ORANGE_BIG': (0, 77, 255), 'UNDEFINED': (170, 160, 154)}
_LEFT, _RIGHT = (255, 232, 56), (162, 95, 255)
_MAX_SIDE = 1920  # output long side; keeps files small and rendering fast


def draw(image: np.ndarray, detections: list[dict], scale: float, kp_conf: float = .5) -> None:
    for d in detections:
        color = _COLORS.get(d.get('class_name'), _COLORS['UNDEFINED'])
        x0, y0, x1, y1 = (np.asarray(d['box_xyxy'], float) * scale)
        bh = max(1., y1 - y0)
        lw = max(1, round(bh * .03))
        cv2.rectangle(image, (round(x0), round(y0)), (round(x1), round(y1)), color, lw, cv2.LINE_AA)
        kp = np.asarray(d.get('keypoints') or [], float)
        n = min(len(kp), 8 if d.get('class_id') == 1 else 6)
        ok = [i < n and np.isfinite(kp[i]).all() and kp[i, 2] > 0 for i in range(n)]
        pts = [(round(kp[i, 0] * scale), round(kp[i, 1] * scale)) if ok[i] else None for i in range(n)]
        for i in range(0, n - 1, 2):
            if pts[i] and pts[i + 1]:
                cv2.line(image, pts[i], pts[i + 1], (255, 255, 255), lw, cv2.LINE_AA)
        r = max(1, round(bh * .035))
        for i, p in enumerate(pts):
            if p:
                fill = (_RIGHT if i % 2 else _LEFT) if kp[i, 2] >= kp_conf else (90, 90, 90)
                cv2.circle(image, p, r, fill, -1, cv2.LINE_AA)
        parts = [d.get('class_name', '?')[0]]
        if d.get('range_m') is not None:
            parts.append(f"{d['range_m']:.1f}m")
        if d.get('landmark_id') is not None:
            parts.append(f"#{d['landmark_id']}")
        text = ' '.join(parts)
        font_scale = max(.25, bh / 110)
        thickness = max(1, round(font_scale * 1.5))
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = round(x0), round(y0) - base - 2
        if ty - th < 0:
            ty = round(y1) + th + 2
        cv2.rectangle(image, (tx, ty - th - 2), (tx + tw + 4, ty + base), (18, 12, 10), -1)
        cv2.putText(image, text, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)


def _encoder_args() -> list[str]:
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    if 'h264_nvenc' in encoders:
        return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23']
    return ['-c:v', 'libopenh264', '-b:v', '12M']


def render(media: Path, result: dict, out: Path) -> Path:
    """Write the annotated media to ``out`` (MP4 for video, JPEG for photo)."""
    kp_conf = float((result.get('options') or {}).get('kp_conf', .5))
    frames = result['frames']
    tmp = out.with_suffix('.part' + out.suffix)
    if result['kind'] == 'photo':
        image = cv2.imread(str(media))
        draw(image, frames[0]['detections'], 1., kp_conf)
        cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tmp.replace(out)
        return out
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required to render annotated video')
    cap = cv2.VideoCapture(str(media))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    scale = min(1., _MAX_SIDE / max(w, h))
    ow, oh = round(w * scale / 2) * 2, round(h * scale / 2) * 2
    by_index = {f['index']: f['detections'] for f in frames}
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{ow}x{oh}', '-r', f'{fps:.6f}', '-i', '-', *_encoder_args(), '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-f', 'mp4', str(tmp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    current, index = [], 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            current = by_index.get(index, current)
            if scale != 1.:
                image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_AREA)
            draw(image, current, scale, kp_conf)
            process.stdin.write(image.tobytes())
            index += 1
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed: ' + process.stderr.read().decode(errors='replace')[-500:])
    except BaseException:
        process.kill()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    tmp.replace(out)
    return out

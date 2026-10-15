"""Native-resolution training tiles of far synthetic cones, plus a replay sample of the real cleaned set.

uv run --locked synth/build_far_dataset.py synth_train [--out datasets/synth-far] [--val-from 2800] [--real 2500]

Frames are rendered for training only (indices 1000+, seeds disjoint from the synth_data test set). Each frame gives
960x600 crops at full sensor resolution, so a cone in a tile has the same pixel size the model sees when it runs on
the full 1920x1200 frame at imgsz 1920:
  far tiles   left and right halves of the horizon band, centred on the cones >= 20 m; kept if they hold one
  near tile   the lower half of the frame at a random x, in one frame of three (keeps near cones in training)
Labels are the postprocess YOLO labels (blue/yellow only), clipped to the tile; cones less than half inside are
dropped and keypoints outside the tile get visibility 0. Frames >= --val-from form the validation split.
The real replay sample comes from the run-D training list, so the model keeps seeing real cones.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / 'datasets/brt-cone-pose'
TW, TH = 960, 600
BY = (0, 4)


def tile_labels(lines, W, H, x0, y0):
    out = []
    for line in lines:
        v = list(map(float, line.split()))
        if int(v[0]) not in BY:
            continue
        cx, cy, w, h = v[1] * W, v[2] * H, v[3] * W, v[4] * H
        a = np.array([cx - w / 2 - x0, cy - h / 2 - y0, cx + w / 2 - x0, cy + h / 2 - y0])
        b = np.clip(a, 0, [TW, TH, TW, TH])
        if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * w * h:
            continue
        row = [int(v[0]), (b[0] + b[2]) / 2 / TW, (b[1] + b[3]) / 2 / TH, (b[2] - b[0]) / TW, (b[3] - b[1]) / TH]
        for k in range(8):
            kx, ky, kv = v[5 + 3 * k:8 + 3 * k]
            px, py = kx * W - x0, ky * H - y0
            if kv > 0 and 0 <= px < TW and 0 <= py < TH:
                row += [px / TW, py / TH, 2]
            else:
                row += [0, 0, 0]
        out.append(' '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out', default=str(ROOT / 'datasets/synth-far'))
    ap.add_argument('--val-from', type=int, default=2800)
    ap.add_argument('--real', type=int, default=2500)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    rng = random.Random(26)
    lists = {'train': [], 'val': []}
    n_cones = {'train': 0, 'val': 0}
    for split in lists:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for ap_ in sorted((src / 'annotations').glob('*.json')):
        a = json.loads(ap_.read_text())
        stem = ap_.stem
        split = 'val' if int(stem) >= args.val_from else 'train'
        W, H = a['width'], a['height']
        img = cv2.imread(str(src / a['image']))
        lines = [l for l in (src / 'labels' / f'{stem}.txt').read_text().splitlines() if l.strip()]
        far = [c for c in a['cones'] if c['class_id'] in BY and c['ground_range_m'] >= 20 and c['visible_pixels'] >= 12]
        tiles = []
        if far:
            ym = np.median([(c['bbox_xyxy_clipped'][1] + c['bbox_xyxy_clipped'][3]) / 2 for c in far])
            y0 = int(np.clip(ym - TH / 2, 0, H - TH)) // 8 * 8
            for x0 in (0, W - TW):
                if any(x0 <= (c['bbox_xyxy_clipped'][0] + c['bbox_xyxy_clipped'][2]) / 2 < x0 + TW for c in far):
                    tiles.append(('far', x0, y0))
        if rng.random() < 1 / 3:
            tiles.append(('near', rng.randrange(0, W - TW + 1, 8), H - TH))
        for kind, x0, y0 in tiles:
            labs = tile_labels(lines, W, H, x0, y0)
            name = f'{stem}_{kind}_{x0}'
            cv2.imwrite(str(out / 'images' / split / f'{name}.jpg'), img[y0:y0 + TH, x0:x0 + TW],
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
            (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(labs) + ('\n' if labs else ''))
            lists[split].append(str(out / 'images' / split / f'{name}.jpg'))
            n_cones[split] += len(labs)
    real = (REAL / 'augmented-by-clean-30k/train.txt').read_text().split()
    rng.shuffle(real)
    real = real[:args.real]
    (out / 'train.txt').write_text('\n'.join(lists['train'] + real) + '\n')
    (out / 'val.txt').write_text('\n'.join(lists['val']) + '\n')
    (out / 'data.yaml').write_text(
        f'path: {out}\ntrain: {out / "train.txt"}\nval: {out / "val.txt"}\nkpt_shape: [8, 3]\n'
        'names:\n  0: BLUE\n  1: ORANGE_BIG\n  2: ORANGE\n  3: UNDEFINED\n  4: YELLOW\n')
    print(f"train: {len(lists['train'])} synth tiles ({n_cones['train']} cones) + {len(real)} real; "
          f"val: {len(lists['val'])} synth tiles ({n_cones['val']} cones)")


if __name__ == '__main__':
    main()

"""Locate the raw source tile for FRAME 0 of the crop TIFF.

Strategy
--------
1. Load only the first slice of the crop (1024x1024 uint8).
2. Iterate every PNG tile (2000x1748 uint8) under the scan_000 section_146 directory.
3. Downsample both crop and each tile by DOWNSAMPLE x, run cv2.matchTemplate.
4. Keep the global best score + its (mfov, tile, position).
5. Refine the top candidate at full resolution to confirm.
"""
import tifffile
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import time

CROP_PATH = Path(r"Z:\Hayworth\Wafer53_SmallVolumeForGlebResTest\PreProcessingSteps\RealOrder_000_slab_146_SmallCropFromCenter.tif")
SRC_DIR   = Path(r"Y:\data\hess_wafer_53\raw\imaging\msem\scan_000\wafer_53_scan_000_20220422_19-52-38\146_")

DOWNSAMPLE = 8
EARLY_STOP_SCORE = 0.95   # bail out once a tile scores this high — clearly the match
PROGRESS_EVERY = 100


def load_crop_frame0():
    with tifffile.TiffFile(CROP_PATH) as tf:
        frame0 = tf.pages[0].asarray()
    print(f"crop frame 0: shape={frame0.shape}, dtype={frame0.dtype}, "
          f"min={frame0.min()}, max={frame0.max()}, mean={frame0.mean():.1f}")
    return frame0


def to_uint8_gray(path):
    """Load a PNG (palette or grayscale) as a 2D uint8 numpy array."""
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return np.asarray(im, dtype=np.uint8)


def downsample(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def match_norm(template_f32, source_f32):
    """Run TM_CCOEFF_NORMED. Returns (best_score, (x, y)) — both in source pixels."""
    if (source_f32.shape[0] < template_f32.shape[0]
            or source_f32.shape[1] < template_f32.shape[1]):
        return -np.inf, (0, 0)
    res = cv2.matchTemplate(source_f32, template_f32, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def stage1_downsampled_sweep(crop_frame, tile_paths, ds):
    crop_ds = downsample(crop_frame, ds).astype(np.float32) / 255.0
    print(f"\nStage 1: downsampled sweep (factor {ds}x). "
          f"Template size {crop_ds.shape}, scanning {len(tile_paths)} tiles...")

    best = {"score": -np.inf, "path": None, "loc_ds": None}
    t0 = time.time()
    for i, path in enumerate(tile_paths):
        if i % PROGRESS_EVERY == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(tile_paths) - i) / rate if rate > 0 else float("inf")
            print(f"  [{i}/{len(tile_paths)}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.1f}/s eta={eta:.0f}s  best={best['score']:.4f}")
        try:
            tile = to_uint8_gray(path)
        except Exception as e:
            print(f"  ERROR reading {path.name}: {e}")
            continue
        tile_ds = downsample(tile, ds).astype(np.float32) / 255.0
        score, loc = match_norm(crop_ds, tile_ds)
        if score > best["score"]:
            best["score"] = score
            best["path"] = path
            best["loc_ds"] = loc
            print(f"  NEW BEST: {path.parent.name}/{path.name}  score={score:.4f}  loc_ds={loc}")
        if score >= EARLY_STOP_SCORE:
            print(f"  Score >= {EARLY_STOP_SCORE} reached — stopping early.")
            break
    print(f"\nStage 1 done in {time.time()-t0:.1f}s. Best score (downsampled): {best['score']:.4f}")
    return best


def stage2_full_res_confirm(crop_frame, best_path, ds_loc, ds):
    print(f"\nStage 2: full-resolution match on {best_path.name}")
    tile = to_uint8_gray(best_path)
    print(f"  tile shape (full): {tile.shape}")

    # Search a window around the downsampled hint to save time.
    # Hint location in full-res pixels, +/- a generous margin:
    hx, hy = ds_loc[0] * ds, ds_loc[1] * ds
    margin = ds * 2     # 16 pixels each side at ds=8
    crop_h, crop_w = crop_frame.shape
    x0 = max(0, hx - margin)
    y0 = max(0, hy - margin)
    x1 = min(tile.shape[1], hx + crop_w + margin)
    y1 = min(tile.shape[0], hy + crop_h + margin)
    sub = tile[y0:y1, x0:x1]
    print(f"  searching window (x,y)=({x0},{y0}) to ({x1},{y1}) shape={sub.shape}")

    sub_f = sub.astype(np.float32) / 255.0
    crop_f = crop_frame.astype(np.float32) / 255.0
    score, loc = match_norm(crop_f, sub_f)
    abs_x = x0 + loc[0]
    abs_y = y0 + loc[1]
    print(f"  full-res score (TM_CCOEFF_NORMED): {score:.6f}")
    print(f"  best location in tile (full-res, top-left): x={abs_x}, y={abs_y}")
    print(f"  -> crop spans x=[{abs_x}, {abs_x+crop_w}], y=[{abs_y}, {abs_y+crop_h}]")

    if score > 0.99:
        print("\n  VERDICT: near-perfect match. Crop frame 0 IS from this tile.")
    elif score > 0.9:
        print("\n  VERDICT: very strong match. Crop frame 0 is from this tile (slight intensity drift).")
    elif score > 0.5:
        print("\n  VERDICT: moderate match. Possibly the right tile but with notable differences.")
    else:
        print("\n  VERDICT: poor match. Crop frame 0 is probably NOT from this tile.")
    return score, (abs_x, abs_y)


def main():
    if not CROP_PATH.exists():
        print(f"crop not found: {CROP_PATH}")
        return
    if not SRC_DIR.exists():
        print(f"source dir not found: {SRC_DIR}")
        return

    frame0 = load_crop_frame0()
    tile_paths = sorted(SRC_DIR.rglob("*.png"))
    print(f"Total tiles under {SRC_DIR}: {len(tile_paths)}")

    best = stage1_downsampled_sweep(frame0, tile_paths, DOWNSAMPLE)
    if best["path"] is None:
        print("No tile yielded a usable match.")
        return

    print(f"\nBest stage-1 candidate: {best['path']}")
    print(f"  mFOV folder       : {best['path'].parent.name}")
    print(f"  tile filename     : {best['path'].name}")
    print(f"  ds match score    : {best['score']:.4f}")

    stage2_full_res_confirm(frame0, best["path"], best["loc_ds"], DOWNSAMPLE)


if __name__ == "__main__":
    main()

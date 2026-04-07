"""
Streaming Pi3X depth prediction from a camera or directory of images.

Usage:
    # From a webcam (device 0):
    python run_stream.py --source camera --camera_id 0

    # From a directory of RGB images (simulates a stream):
    python run_stream.py --source dir --rgb_dir /path/to/rgb

    # From a video file:
    python run_stream.py --source video --video_path /path/to/video.mp4

    # With custom intrinsics:
    python run_stream.py --source camera --fx 800 --fy 800 --cx 320 --cy 240
"""

import argparse
import time
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo_stream import Pi3XVOStream


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_PI3X_CKPT = REPO_ROOT / "chkp" / "Pi3X" / "model.safetensors"

# ──────────────────────────────────────────────────────────────
#  Image preprocessing (matches pi3/utils/basic.py logic)
# ──────────────────────────────────────────────────────────────

def compute_target_size(w_orig, h_orig, pixel_limit=255000):
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1
    w_t, h_t = w_orig * scale, h_orig * scale
    k, m = round(w_t / 14), round(h_t / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_t / h_t:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def preprocess_frame(bgr_frame, target_w, target_h, to_tensor):
    """BGR numpy (H,W,3) uint8 -> (3, target_h, target_w) float32 tensor in [0,1]."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return to_tensor(rgb_resized)


def build_intrinsics(fx, fy, cx, cy, orig_w, orig_h, target_w, target_h, device):
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    K = torch.tensor([
        [fx * scale_x, 0.0, cx * scale_x],
        [0.0, fy * scale_y, cy * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    return K.view(1, 1, 3, 3)


# ──────────────────────────────────────────────────────────────
#  Frame sources
# ──────────────────────────────────────────────────────────────

def camera_source(camera_id=0):
    """Yields BGR frames from a camera."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    print(f"[source] Camera {camera_id} opened")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def directory_source(rgb_dir):
    """Yields BGR frames from a sorted directory of images."""
    exts = {'.png', '.jpg', '.jpeg'}
    files = sorted(
        p for p in Path(rgb_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if not files:
        raise ValueError(f"No images found in {rgb_dir}")
    print(f"[source] Directory: {rgb_dir} ({len(files)} images)")
    for f in files:
        frame = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if frame is not None:
            yield frame


def video_source(video_path):
    """Yields BGR frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    print(f"[source] Video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────
#  Result handler (save / display / etc.)
# ──────────────────────────────────────────────────────────────

def handle_results(results, output_dir, frame_offset):
    """Save depth maps and poses from a chunk result."""
    depth = results['depth'].cpu().numpy()   # (n, H, W)
    poses = results['poses'].cpu().numpy()   # (n, 4, 4)
    conf = results['conf'].cpu().numpy()     # (n, H, W)
    n = depth.shape[0]

    for i in range(n):
        idx = frame_offset + i

        # Save depth as uint16 PNG (millimeters)
        d = depth[i]
        valid = np.logical_and(d > 0, np.isfinite(d))
        d_u16 = np.zeros_like(d, dtype=np.uint16)
        if valid.sum() > 0:
            q = np.round(d[valid] / 0.001)  # 1 mm scale
            q = np.clip(q, 1, 65535).astype(np.uint16)
            d_u16[valid] = q
        cv2.imwrite(str(output_dir / f'depth{idx:06d}.png'), d_u16)

    # Append poses to trajectory file
    traj_path = output_dir / 'trajectory.txt'
    with open(traj_path, 'a') as f:
        for i in range(n):
            f.write(' '.join(map(str, poses[i].flatten())) + '\n')

    print(f"  Saved frames {frame_offset}..{frame_offset + n - 1}")
    return n


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Streaming Pi3X depth prediction")
    parser.add_argument("--source", choices=["camera", "dir", "video"], default="dir")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--rgb_dir", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_stream")
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--fx", type=float, default=800.0)
    parser.add_argument("--fy", type=float, default=800.0)
    parser.add_argument("--cx", type=float, default=640.0)
    parser.add_argument("--cy", type=float, default=360.0)
    args = parser.parse_args()

    # Output directory
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Clear trajectory file
    (out / 'trajectory.txt').write_text('')

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    print("Loading Pi3X model...")
    if LOCAL_PI3X_CKPT.is_file():
        model = Pi3X().to(device).eval()
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(LOCAL_PI3X_CKPT)), strict=False)
    else:
        model = Pi3X.from_pretrained('yyfz233/Pi3X').to(device).eval()
    print("Model loaded.")

    # Get first frame to determine resolution
    if args.source == "camera":
        source_iter = camera_source(args.camera_id)
    elif args.source == "dir":
        if args.rgb_dir is None:
            raise ValueError("--rgb_dir required for dir source")
        source_iter = directory_source(args.rgb_dir)
    elif args.source == "video":
        if args.video_path is None:
            raise ValueError("--video_path required for video source")
        source_iter = video_source(args.video_path)

    # Peek first frame for resolution
    source_iter = iter(source_iter)
    first_frame = next(source_iter)
    orig_h, orig_w = first_frame.shape[:2]
    target_w, target_h = compute_target_size(orig_w, orig_h)
    print(f"Original: {orig_w}x{orig_h} -> Inference: {target_w}x{target_h}")

    # Build intrinsics
    intrinsics = build_intrinsics(
        args.fx, args.fy, args.cx, args.cy,
        orig_w, orig_h, target_w, target_h, device
    )
    print(f"Scaled intrinsics:\n{intrinsics[0, 0].cpu().numpy()}")

    # Create streaming pipeline
    stream = Pi3XVOStream(
        model=model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        conf_thre=0.05,
        intrinsics=intrinsics,
        dtype=dtype,
    )

    to_tensor = transforms.ToTensor()
    frame_offset = 0
    total_frames = 0
    t_start = time.perf_counter()

    # Process first frame
    frame_tensor = preprocess_frame(first_frame, target_w, target_h, to_tensor).to(device)
    results = stream.push_frame(frame_tensor)
    total_frames += 1
    if results is not None:
        n = handle_results(results, out, frame_offset)
        frame_offset += n

    # Process remaining frames
    for bgr_frame in source_iter:
        frame_tensor = preprocess_frame(bgr_frame, target_w, target_h, to_tensor).to(device)
        results = stream.push_frame(frame_tensor)
        total_frames += 1

        if results is not None:
            n = handle_results(results, out, frame_offset)
            frame_offset += n

    # Flush remaining
    results = stream.flush()
    if results is not None:
        n = handle_results(results, out, frame_offset)
        frame_offset += n

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. Processed {total_frames} frames in {elapsed:.1f}s")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()

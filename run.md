# run.py — Pi3X Depth & Pose Pipeline

## Overview

`run.py` runs the Pi3X depth estimation and visual odometry pipeline on one or more datasets. Each dataset is defined by a YAML config file in the `configs/` directory.

## Usage

```bash
# Run on a single dataset (by config name)
python run.py isaac_sim

# Run on multiple datasets
python run.py isaac_sim scannetpp

# Pass full path to config
python run.py configs/isaac_sim.yaml

# Keep original image resolution instead of downscaling
python run.py isaac_sim --original_img

# dynamic chunk size, overlap
python run.py isaac_sim --chunk_size 50 --overlap 15
```

## Project Structure

```
.
├── run.py                  # Entry point — loads configs, iterates scenes
├── utils.py                # Core logic — model loading, inference, saving
├── configs/
│   ├── isaac_sim.yaml      # Isaac Sim dataset config
│   └── scannetpp.yaml      # ScanNet++ dataset config
└── chkp/
    └── Pi3X/
        └── model.safetensors   # Local Pi3X checkpoint (required)
```

## Config File Format

Each YAML config has three sections:

```yaml
dataset: isaac_sim              # Dataset identifier (for logging)

scenes:                         # List of scene directories to process
  - /path/to/scene_1
  - /path/to/scene_2

camera:                         # Pinhole intrinsics (in original image resolution)
  fx: 800.0
  fy: 800.0
  cx: 640.0
  cy: 360.0

pi3:                            # Pi3X model parameters
  depth_model: yyfz233/Pi3X     # Model name
  chunk_size: 30                # Frames per inference chunk
  overlap: 10                   # Overlap between consecutive chunks
  pi3_png_depth_scale: 0.001    # Depth encoding scale (meters per uint16 unit)
```

### Config Fields

| Field | Description |
|---|---|
| `scenes` | Each scene directory must contain an `rgb/` folder with images and a `depth/` folder |
| `camera.fx/fy/cx/cy` | Camera intrinsics in the original RGB image resolution. Optional — Pi3X can run without them, but providing them gives metrically accurate depth |
| `pi3.chunk_size` | Number of frames processed together in one forward pass. Larger = more context but more GPU memory |
| `pi3.overlap` | Number of frames shared between consecutive chunks for alignment. Must be < `chunk_size` |
| `pi3.pi3_png_depth_scale` | Encoding scale for saved depth PNGs. `0.001` means 1 unit = 1 mm, giving 0–65.535 m range |

## Expected Scene Directory Layout

```
scene_dir/
├── rgb/                    # Input: RGB images (png/jpg)
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/                  # Input: original depth (not used by Pi3X, path tracked for reference)
└── camera_poses.txt        # Input: original trajectory (not used by Pi3X, path tracked for reference)
```

## Output

For a run with `chunk_size=30` and `overlap=10`, the pipeline creates:

```
scene_dir/
├── pi3_depth_30_10/        # Output: Pi3X predicted depth maps
│   ├── depth000000.png     #   uint16 PNGs (value × pi3_png_depth_scale = depth in meters)
│   ├── depth000001.png
│   ├── ...
│   ├── depth_video.mp4     #   Grayscale depth visualization video
│   └── pi3_depth_meta.txt  #   Encoding metadata (scale, format, depth range)
└── pi3_traj_30_10.txt      # Output: camera poses (N lines, 16 floats = flattened 4×4 matrix)
```

The output folder/file names encode the chunk parameters: `pi3_depth_{chunk_size}_{overlap}` and `pi3_traj_{chunk_size}_{overlap}.txt`. This lets you compare results from different chunk/overlap settings side by side.

## How It Works

1. **Config loading** — `run.py` loads each YAML config via OmegaConf
2. **Scene iteration** — For each scene in the config, builds a per-scene config with paths, camera params, and pi3 settings
3. **`process_depth_model()`** in `utils.py` handles the rest:
   - Loads Pi3X model from local checkpoint (`chkp/Pi3X/model.safetensors`)
   - Loads RGB images as tensors, optionally resizing them
   - Scales camera intrinsics to match the inference resolution
   - Runs `Pi3XVO` pipeline with the specified `chunk_size` and `overlap`
   - Extracts per-frame depth maps from predicted 3D points
   - Upscales depth maps back to original RGB resolution
   - Saves depth as uint16 PNGs + visualization video
   - Saves camera poses as a text file

## Checkpoint

The local checkpoint is expected at `chkp/Pi3X/model.safetensors`. If `USE_LOCAL_CKPT_ONLY=True` (default), the script fails immediately if this file is missing. Set it to `False` to fall back to downloading from Hugging Face.

---

### run stream
```
# Live camera with 30-frame max visible clouds
python run_stream.py --source camera --max_pcds 30

# Directory (simulated stream)
python run_stream.py --source dir --rgb_dir /path/to/rgb --max_pcds 50 --subsample 4

# Video, also save depth PNGs
python run_stream.py --source video --video_path scene.mp4 --output_dir out/ --max_pcds 40

# No visualization (save only, like before)
python run_stream.py --source dir --rgb_dir /path --no_vis --output_dir out/

```

### running with intrinsics
```
python run.py isaac_sim

# original-size mode
python run.py isaac_sim --original_img

# multiple datasets
python run.py isaac_sim scannetpp
```

```
python visualize_gt_vs_pi3x.py --scene_dir /home/maribjonov_mr/IsaacSim_bench/scene_2 --accumulate 29
```

### aligning pi3 to gt pcds
```
python align_pi3_gt.py \
  --scene_dir /home/maribjonov_mr/IsaacSim_bench/scene_2 \
  --save_transform

# saved automatically in scene folder:
#   pi3_to_world_transform.json
#   pi3_to_world_transform.npz

  # if memory is issue
  python align_pi3_gt.py --scene_dir ... --align_subsample 12 --max_pairs 150000 --frame_pair_cap 3000

  # saving align transform
   python align_pi3_gt.py --scene_dir /home/yehia/rizo/IsaacSim_bench_pi3/scene_3 --max_pairs 250000 --save_transform

   # visualizing only pi3 pcds
   python align_pi3_gt.py --scene_dir /home/yehia/rizo/IsaacSim_bench_pi3/scene_3 --pi3_only
```

#### aligning matrices
**scene 2 16-12**:
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.1704333623
Rotation R:
[[-0.83297632  0.55292118 -0.02070306]
 [ 0.36008542  0.51330042 -0.77901294]
 [-0.42010587 -0.65635421 -0.62666595]]
Translation t:
[4.84204982 4.33227449 1.34055737]
Alignment RMSE (m): 0.285196
Sim(3) matrix:
[[-0.97494328  0.6471574  -0.02423155  4.84204982]
 [ 0.42145599  0.60078394 -0.91178274  4.33227449]
 [-0.49170592 -0.76821886 -0.73347074  1.34055737]
 [ 0.          0.          0.          1.        ]]
```

**scene 2 full**:
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.1598583531
Rotation R:
[[-0.87257264  0.48159678  0.0817406 ]
 [ 0.248681    0.58197978 -0.77424627]
 [-0.42044589 -0.65525878 -0.62758361]]
Translation t:
[6.26369957 4.58098195 1.34531721]
Alignment RMSE (m): 0.283764
Sim(3) matrix:
[[-1.01206066  0.55858405  0.09480752  6.26369957]
 [ 0.28843473  0.67501411 -0.89801601  4.58098195]
 [-0.48765768 -0.76000737 -0.72790809  1.34531721]
 [ 0.          0.          0.          1.        ]]
```
**scene cabinet_simple 60 frames, 16-12**
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.1365642071
Rotation R:
[[-0.9094795   0.28849628  0.29936087]
 [-0.01619221  0.69492627 -0.71889867]
 [-0.41543333 -0.65867092 -0.6273498 ]]
Translation t:
[-0.34668418  0.23423761  1.30144032]
Alignment RMSE (m): 0.094035
Sim(3) matrix:
[[-1.03368185  0.32789455  0.34024285 -0.34668418]
 [-0.01840349  0.78982832 -0.81707449  0.23423761]
 [-0.47216665 -0.74862179 -0.71302333  1.30144032]
 [ 0.          0.          0.          1.        ]]
```
**scene cabinet_simple 60 frames, 30-12**
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.0906889340
Rotation R:
[[-0.83284226  0.00201374  0.55350675]
 [-0.3620696   0.75439226 -0.54753806]
 [-0.41866381 -0.65642079 -0.62756064]]
Translation t:
[0.05804596 0.17697884 1.29603352]
Alignment RMSE (m): 0.092250
Sim(3) matrix:
[[-0.90837183  0.00219636  0.60370368  0.05804596]
 [-0.3949053   0.82280729 -0.5971937   0.17697884]
 [-0.45663198 -0.7159509  -0.68447345  1.29603352]
 [ 0.          0.          0.          1.        ]]
```

### how depth saving works (Pi3 output)
- Pi3 predicts per-pixel depth in meters: `D_m(u,v)`.
- We save metric PNGs as `uint16` (not normalized uint8).
- Encoding uses a fixed scale `s = pi3_png_depth_scale` (default `0.001` m/unit):

```text
q(u,v) = round(D_m(u,v) / s)          # quantized depth value
q(u,v) is clipped to [1, 65535] for valid pixels, 0 for invalid
```

- Saved file value is `q(u,v)` in `depth%06d.png`.
- A metadata file `pi3_depth_meta.txt` stores `png_depth_scale`, so decoding is deterministic:

```text
D_m(u,v) = q(u,v) * s
```

### how depth -> point cloud reconstruction works
For each valid depth pixel `(u, v, z)` with `z = D_m(u,v)` and intrinsics `(fx, fy, cx, cy)`:

```text
x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
z_cam = z
```

This gives camera-frame 3D point:

```text
p_cam = [x_cam, y_cam, z_cam]^T
```

Then transform to world frame using camera-to-world pose `T_c2w = [R|t]`:

```text
p_world = R * p_cam + t
```

All `p_world` points are accumulated into the Open3D point cloud.

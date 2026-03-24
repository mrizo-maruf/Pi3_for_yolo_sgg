### running with intrinsics
```
python run.py

# original-size mode
python run.py --original_img

```

```
python visualize_gt_vs_pi3x.py --scene_dir /home/maribjonov_mr/IsaacSim_bench/scene_2 --accumulate 29
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

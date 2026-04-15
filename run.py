from utils import process_depth_model
import argparse
from pathlib import Path
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = REPO_ROOT / "configs"
LOCAL_PI3X_CKPT = REPO_ROOT / "chkp" / "Pi3X" / "model.safetensors"
USE_LOCAL_CKPT_ONLY = True

if USE_LOCAL_CKPT_ONLY and not LOCAL_PI3X_CKPT.is_file():
    raise FileNotFoundError(
        f"Local Pi3X checkpoint not found at: {LOCAL_PI3X_CKPT}\n"
        "Either place model.safetensors there or set USE_LOCAL_CKPT_ONLY=False."
    )


def main():
    parser = argparse.ArgumentParser(description="Run Pi3X depth + pose processing on dataset scenes.")
    parser.add_argument(
        "configs",
        nargs="+",
        help="Dataset config YAML files (name without .yaml or full path). "
             "E.g.: isaac_sim scannetpp  or  configs/isaac_sim.yaml",
    )
    parser.add_argument(
        "--original_img",
        action="store_true",
        help="Use original image resolution (with minimal padding to meet model constraints).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Override chunk_size from config.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Override overlap from config.",
    )
    args = parser.parse_args()

    for config_arg in args.configs:
        config_path = Path(config_arg)
        if not config_path.suffix:
            config_path = CONFIGS_DIR / f"{config_arg}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")

        dataset_cfg = OmegaConf.load(config_path)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_cfg.get('dataset', config_path.stem)}")
        print(f"Config:  {config_path}")
        print(f"{'='*60}")

        global_cam = dataset_cfg.get("camera", None)
        pi3 = dataset_cfg.pi3
        scenes = dataset_cfg.get("scenes", [])

        for scene_entry in scenes:
            # Scenes can be plain strings (use global camera) or dicts with per-scene camera
            if isinstance(scene_entry, str):
                scene_path = scene_entry
                cam = global_cam
            else:
                scene_path = scene_entry.path
                cam = scene_entry.get("camera", global_cam)

            if cam is None:
                raise ValueError(f"No camera intrinsics for scene: {scene_path}. "
                                 "Provide per-scene camera or a global camera section.")

            rgb_dir = str(Path(scene_path) / "images")
            depth_dir = str(Path(scene_path) / "gt_depth")

            temp_cfg = OmegaConf.create({
                "rgb_dir": rgb_dir,
                "depth_dir": depth_dir,
                "traj_path": str(Path(scene_path) / "camera_poses.txt"),
                "depth_model": pi3.depth_model,
                "ckpt": str(LOCAL_PI3X_CKPT) if LOCAL_PI3X_CKPT.is_file() else None,
                "original_img": args.original_img,
                "pi3_png_depth_scale": pi3.pi3_png_depth_scale,
                "chunk_size": args.chunk_size if args.chunk_size is not None else pi3.chunk_size,
                "overlap": args.overlap if args.overlap is not None else pi3.overlap,
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
            })
            process_depth_model(temp_cfg)


if __name__ == "__main__":
    main()

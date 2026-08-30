#!/usr/bin/env python3
"""Validate the converted single-arm UMI dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


register_codecs()


REQUIRED_DATA_KEYS = {
    "action",
    "camera0_rgb",
    "robot0_demo_end_pose",
    "robot0_demo_start_pose",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
}


class DatasetValidationError(ValueError):
    """Raised when a converted dataset cannot feed the training config."""


def validate_dataset(path: str | Path, expected_image_size: int = 224) -> dict[str, Any]:
    """Validate storage keys, time alignment and training-facing dimensions."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise DatasetValidationError(f"dataset does not exist: {dataset_path}")
    if dataset_path.suffix != ".zip":
        raise DatasetValidationError("the supported UMI dataset artifact is a .zarr.zip file")

    try:
        with zarr.ZipStore(str(dataset_path), mode="r") as store:
            root = zarr.group(store=store)
            if "data" not in root or "meta" not in root:
                raise DatasetValidationError("dataset must contain data/ and meta/ groups")
            data = root["data"]
            missing = sorted(REQUIRED_DATA_KEYS.difference(data.array_keys()))
            if missing:
                raise DatasetValidationError(f"dataset is missing arrays: {', '.join(missing)}")
            if "episode_ends" not in root["meta"]:
                raise DatasetValidationError("dataset is missing meta/episode_ends")

            episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            if episode_ends.ndim != 1 or episode_ends.size == 0:
                raise DatasetValidationError("meta/episode_ends must contain at least one episode")
            if np.any(np.diff(np.concatenate(([0], episode_ends))) <= 0):
                raise DatasetValidationError("episode ends must be strictly increasing")
            num_steps = int(episode_ends[-1])

            shapes = {name: tuple(data[name].shape) for name in REQUIRED_DATA_KEYS}
            for name, shape in shapes.items():
                if not shape or shape[0] != num_steps:
                    raise DatasetValidationError(
                        f"{name} has {shape[0] if shape else 'no'} steps; expected {num_steps}"
                    )
            if shapes["action"][-1] != 7:
                raise DatasetValidationError(
                    "raw action must have 7 values (position 3 + axis-angle 3 + gripper 1); "
                    f"got {shapes['action'][-1]}"
                )
            expected_dims = {
                "robot0_eef_pos": 3,
                "robot0_eef_rot_axis_angle": 3,
                "robot0_gripper_width": 1,
                "robot0_demo_start_pose": 6,
                "robot0_demo_end_pose": 6,
            }
            for name, width in expected_dims.items():
                if len(shapes[name]) != 2 or shapes[name][-1] != width:
                    raise DatasetValidationError(f"{name} must have shape [T, {width}]")
            image_shape = shapes["camera0_rgb"]
            if image_shape[1:] != (expected_image_size, expected_image_size, 3):
                raise DatasetValidationError(
                    "camera0_rgb must have shape "
                    f"[T, {expected_image_size}, {expected_image_size}, 3]; got {image_shape}"
                )

            for name in REQUIRED_DATA_KEYS - {"camera0_rgb"}:
                if not np.isfinite(data[name][:]).all():
                    raise DatasetValidationError(f"{name} contains NaN or infinite values")

            return {
                "dataset": str(dataset_path),
                "num_episodes": int(episode_ends.size),
                "num_steps": num_steps,
                "image_size": expected_image_size,
                "raw_action_dim": 7,
                "training_action_dim": 10,
                "arrays": {name: list(shapes[name]) for name in sorted(shapes)},
            }
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(f"cannot open {dataset_path} as a UMI zarr zip: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    try:
        report = validate_dataset(args.dataset, expected_image_size=args.image_size)
    except DatasetValidationError as exc:
        raise SystemExit(f"dataset validation error: {exc}") from exc
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

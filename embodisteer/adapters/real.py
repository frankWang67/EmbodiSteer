"""Tensor adaptation for real-world inference; configuration stays lightweight."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from embodisteer.real_config import (
    load_obstacle_geometry, load_yaml_mapping, validate_robot_config,
)

if TYPE_CHECKING:
    import torch


def load_obstacle_config(path: str | Path | None) -> list[dict[str, torch.Tensor]]:
    geometry = load_obstacle_geometry(path)
    import torch

    return [
        {
            "center": torch.tensor([entry["center"]], dtype=torch.float32),
            "quat": torch.tensor([entry["quat_wxyz"]], dtype=torch.float32),
            "extent": torch.tensor([entry["half_extent"]], dtype=torch.float32),
        }
        for entry in geometry
    ]


__all__ = ["load_obstacle_config", "load_yaml_mapping", "validate_robot_config"]

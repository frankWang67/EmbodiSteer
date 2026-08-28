#!/usr/bin/env python3
"""Static checks for the isolated EmbodiSteer release tree.

The default mode does not import CUDA, ManiSkill or cuRobo. Use
``--import-policy`` only in an environment where those dependencies are
installed.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from embodisteer.runtime import repository_root, third_party_manifest_path


REQUIRED_PATHS = (
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "README.md",
    "embodisteer/__init__.py",
    "embodisteer/policies/ee2joint.py",
    "embodisteer/policies/ee_space.py",
    "embodisteer/policies/baselines.py",
    "embodisteer/policies/jm2d.py",
    "embodisteer/kinematics/pose.py",
    "embodisteer/collision/sdf.py",
    "embodisteer/guidance/cbf.py",
    "embodisteer/guidance/schedule.py",
    "embodisteer/guidance/diffusion.py",
    "embodisteer/kinematics/rotation.py",
    "embodisteer/runtime_config.py",
    "configs/policy/embodisteer.yaml",
    "configs/policy/ee.yaml",
    "eval_sim_single_robot.py",
    "eval_sim_multi_robots.py",
    "eval_real.py",
    "third_party/manifest.yaml",
    "third_party/assets.yaml",
    "THIRD_PARTY_NOTICES.md",
    "artifacts/manifest.yaml",
    "environment/environment.yaml",
    "environment/requirements.txt",
    "environment/dependencies.md",
    "configs/experiments/paper.yaml",
    "configs/real/robot.template.yaml",
    "docs/installation.md",
    "docs/method.md",
    "docs/policy_configuration.md",
    "docs/architecture.md",
    "docs/simulation.md",
    "docs/real_world.md",
    "docs/data_and_checkpoints.md",
    "docs/paper_reproduction.md",
)

REQUIRED_ARTIFACT_FIELDS = {
    "id",
    "type",
    "task",
    "revision",
    "sha256",
    "size_bytes",
    "url",
    "license",
    "access",
}

FORBIDDEN_PROJECT_FILES_IN_UMI = (
    "diffusion_policy/common/guided_diffusion_util.py",
    "diffusion_policy/common/rotation_conversions.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_ee2joint_space.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_ee_space.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_baseline.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_jm2d.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_joint_space.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_joint_space_with_guidance.py",
    "diffusion_policy/policy/diffusion_unet_timm_policy_with_guidance.py",
)

TEXT_SUFFIXES = {".cff", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
PUBLIC_STRING_PATTERNS = {
    "private IPv4 address": re.compile(
        r"(?<![0-9.])(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
        r"192\.168\.(?:\d{1,3}\.)\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})(?![0-9.])"
    ),
    "author-local hostname": re.compile(r"internal\.tri\.global", re.IGNORECASE),
    "author-local path": re.compile(r"/(?:home|data)/wshf(?:/|\b)"),
    "author-private registry": re.compile(r"wshf" + r"-docker-mirror", re.IGNORECASE),
}


def check_public_strings(root: Path) -> list[str]:
    """Reject site-specific addresses and author-local paths in release text."""
    errors: list[str] = []
    ignored_parts = {
        ".codex",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative_path = path.relative_to(root)
        if ignored_parts.intersection(relative_path.parts):
            continue
        # The pinned forks are external dependencies materialized into this
        # gitignored directory. Their upstream examples are not release text
        # owned by this repository and must not affect the public-tree scan.
        if relative_path.parts[:2] == ("third_party", "src"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PUBLIC_STRING_PATTERNS.items():
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative_path}:{line_number}: {label}")
    return errors


def load_yaml_mapping(path: Path) -> tuple[dict, list[str]]:
    """Load a YAML document and report parse/type failures as layout errors."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {}, [f"{path.relative_to(REPO_ROOT)}: cannot read YAML: {exc}"]
    except yaml.YAMLError as exc:
        return {}, [f"{path.relative_to(REPO_ROOT)}: invalid YAML: {exc}"]
    if not isinstance(value, dict):
        return {}, [f"{path.relative_to(REPO_ROOT)}: expected a YAML mapping"]
    return value, []


def check_publication_metadata(root: Path) -> list[str]:
    errors: list[str] = []

    citation, citation_errors = load_yaml_mapping(root / "CITATION.cff")
    errors.extend(citation_errors)
    if citation:
        if str(citation.get("cff-version")) != "1.2.0":
            errors.append("CITATION.cff: cff-version must be 1.2.0")
        for field in ("message", "title", "authors"):
            if not citation.get(field):
                errors.append(f"CITATION.cff: {field} is missing")
        if citation.get("type") != "software":
            errors.append("CITATION.cff: type must be software")

    artifacts, artifact_errors = load_yaml_mapping(root / "artifacts/manifest.yaml")
    errors.extend(artifact_errors)
    if artifacts:
        if artifacts.get("schema_version") != 1:
            errors.append("artifacts/manifest.yaml: unsupported schema_version")
        if artifacts.get("status") != "metadata_only":
            errors.append("artifacts/manifest.yaml: expected metadata_only status")
        publication = artifacts.get("publication")
        if not isinstance(publication, dict):
            errors.append("artifacts/manifest.yaml: publication must be a mapping")
        else:
            for field in ("checkpoints", "training_data"):
                if publication.get(field) != "not_published":
                    errors.append(
                        f"artifacts/manifest.yaml: publication.{field} must be "
                        "not_published for this release"
                    )
        if not isinstance(artifacts.get("artifacts"), list):
            errors.append("artifacts/manifest.yaml: artifacts must be a list")
        declared_fields = set(artifacts.get("required_fields_when_published", []))
        missing_fields = REQUIRED_ARTIFACT_FIELDS - declared_fields
        if missing_fields:
            errors.append(
                "artifacts/manifest.yaml: missing future artifact fields: "
                + ", ".join(sorted(missing_fields))
            )

    assets, asset_errors = load_yaml_mapping(root / "third_party/assets.yaml")
    errors.extend(asset_errors)
    if assets:
        if assets.get("schema_version") != 1:
            errors.append("third_party/assets.yaml: unsupported schema_version")
        groups = assets.get("asset_groups")
        if not isinstance(groups, dict) or not groups:
            errors.append("third_party/assets.yaml: asset_groups must be non-empty")
        else:
            for name, spec in groups.items():
                if not isinstance(spec, dict):
                    errors.append(f"third_party/assets.yaml: {name} must be a mapping")
                    continue
                for field in (
                    "path",
                    "source_repository",
                    "source_revision",
                    "license",
                    "redistribution_status",
                    "review_status",
                ):
                    if not spec.get(field):
                        errors.append(f"third_party/assets.yaml: {name}.{field} is missing")
                if spec.get("redistribution_status") == "included":
                    asset_path = root / str(spec.get("path", ""))
                    if not asset_path.exists():
                        errors.append(
                            f"third_party/assets.yaml: included path does not exist: "
                            f"{spec.get('path')}"
                        )
                    evidence = spec.get("license_evidence")
                    if not isinstance(evidence, list) or not evidence:
                        errors.append(
                            f"third_party/assets.yaml: {name}.license_evidence is missing"
                        )
                    else:
                        for evidence_path in evidence:
                            if not (root / str(evidence_path)).is_file():
                                errors.append(
                                    "third_party/assets.yaml: license evidence does not "
                                    f"exist: {evidence_path}"
                                )

    return errors


def check_layout(root: Path) -> list[str]:
    errors = [name for name in REQUIRED_PATHS if not (root / name).is_file()]
    errors.extend(
        f"project-specific file must live under embodisteer/: {name}"
        for name in FORBIDDEN_PROJECT_FILES_IN_UMI
        if (root / name).exists()
    )
    manifest, manifest_errors = load_yaml_mapping(third_party_manifest_path())
    errors.extend(manifest_errors)
    if manifest:
        if manifest.get("schema_version") != 1:
            errors.append("third_party/manifest.yaml: unsupported schema_version")
        for name, spec in manifest.get("repositories", {}).items():
            for field in ("upstream_url", "fork_url", "status", "install"):
                if not spec.get(field):
                    errors.append(f"third_party/manifest.yaml: {name}.{field} is missing")
    errors.extend(check_publication_metadata(root))
    errors.extend(check_public_strings(root))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-policy", action="store_true")
    args = parser.parse_args()

    root = repository_root()
    errors = check_layout(root)
    if args.import_policy:
        importlib.import_module("embodisteer.policies.ee2joint")
        importlib.import_module("embodisteer.policies.ee_space")
        importlib.import_module("embodisteer.policies.baselines")
        importlib.import_module("embodisteer.policies.jm2d")
        importlib.import_module("embodisteer.policies.legacy")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"EmbodiSteer layout OK: {root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate or materialize the pinned third-party repositories.

The command is intentionally opt-in: importing the project never clones or
installs anything. A null commit is rejected unless ``--allow-pending`` is
used for local inspection, because a dirty dependency must not become an
accidental release input.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from embodisteer.runtime import repository_root

PROFILE_REPOSITORIES = {
    "simulation": ("maniskill", "curobo"),
    "real": ("curobo",),
    "all": ("maniskill", "curobo"),
}


def _run(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        command = " ".join(args)
        raise SystemExit(f"command failed: {command}\n{details}".rstrip()) from exc
    return result.stdout.strip()


def _load_manifest() -> dict[str, Any]:
    path = repository_root() / "third_party" / "manifest.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_manifest(allow_pending: bool = False, profile: str = "all") -> list[str]:
    manifest = _load_manifest()
    errors: list[str] = []
    for name in PROFILE_REPOSITORIES[profile]:
        spec = manifest.get("repositories", {}).get(name, {})
        for field in ("fork_url", "upstream_url", "install", "status"):
            if not spec.get(field):
                errors.append(f"{name}: missing manifest field {field}")
        if spec.get("commit") is None and not allow_pending:
            errors.append(f"{name}: commit is pending; create a clean pinned commit first")
    return errors


def materialize(
    allow_pending: bool = False,
    install: bool = False,
    cuda_home: Path | None = None,
    profile: str = "all",
) -> None:
    errors = validate_manifest(allow_pending=allow_pending, profile=profile)
    if errors:
        raise SystemExit("\n".join(errors))

    manifest = _load_manifest()
    root = repository_root()
    for name in PROFILE_REPOSITORIES[profile]:
        spec = manifest["repositories"][name]
        commit = spec.get("commit")
        if commit is None:
            print(f"SKIP {name}: pending clean commit")
            continue

        destination = root / "third_party" / "src" / name
        if destination.exists():
            current = _run(["git", "rev-parse", "HEAD"], cwd=destination)
            if current != commit:
                raise SystemExit(
                    f"{name}: {destination} exists at {current}, expected {commit}; "
                    "move it aside or inspect it manually"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--no-checkout", spec["fork_url"], str(destination)])
            _run(["git", "checkout", "--detach", commit], cwd=destination)

        print(f"READY {name}: {commit}")
        if install:
            install_args = [sys.executable, "-m", "pip", "install"]
            install_env = None
            if name == "curobo":
                # cuRobo imports torch while defining its CUDA extensions.
                # Build isolation may download a different torch/CUDA build,
                # so compile against the already pinned runtime environment.
                install_args.append("--no-build-isolation")
                if cuda_home is not None:
                    nvcc = cuda_home / "bin" / "nvcc"
                    if not nvcc.is_file():
                        raise SystemExit(f"CUDA toolkit compiler not found: {nvcc}")
                    install_env = os.environ.copy()
                    install_env["CUDA_HOME"] = str(cuda_home)
            install_args.extend(["-e", str(destination)])
            _run(install_args, env=install_env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_REPOSITORIES, default="all",
                        help="simulation: ManiSkill + cuRobo; real: cuRobo only; all: compatibility default")
    parser.add_argument("--check", action="store_true", help="validate only")
    parser.add_argument("--allow-pending", action="store_true", help="allow null commits for inspection")
    parser.add_argument("--install", action="store_true", help="pip install each materialized checkout")
    parser.add_argument(
        "--cuda-home",
        type=Path,
        help="CUDA toolkit root used to compile the pinned cuRobo extensions",
    )
    args = parser.parse_args()

    errors = validate_manifest(allow_pending=args.allow_pending, profile=args.profile)
    if args.check or errors:
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"third_party/manifest.yaml: OK (profile={args.profile})")
        return
    materialize(
        allow_pending=args.allow_pending,
        install=args.install,
        cuda_home=args.cuda_home,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()

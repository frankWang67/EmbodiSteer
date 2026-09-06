from pathlib import Path

from scripts.diagnostics.check_release_layout import (
    RETIRED_POLICY_FILES,
    check_layout,
    check_public_strings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_tree_is_self_contained_at_top_level():
    required = (
        "CITATION.cff",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "embodisteer",
        "diffusion_policy",
        "umi",
        "configs/policy/embodisteer.yaml",
        "third_party/manifest.yaml",
        "third_party/assets.yaml",
        "THIRD_PARTY_NOTICES.md",
        "docs/method.md",
        "run_sim_pipeline.sh",
        "run_sim_workflow.py",
        "eval_sim_single_robot.py",
        "eval_sim_multi_robots.py",
        "pyproject.toml",
        "environment/requirements.txt",
        "environment/dependencies.md",
    )
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"missing release paths: {missing}"


def test_release_layout_checks_pass_for_current_tree():
    assert check_layout(ROOT) == []


def test_retired_policy_implementations_are_absent():
    restored = [name for name in RETIRED_POLICY_FILES if (ROOT / name).exists()]
    assert not restored, f"retired policy paths were restored: {restored}"


def test_public_string_scan_ignores_local_runtime_artifacts(tmp_path):
    local_metadata = tmp_path / "data_local" / "run" / ".hydra" / "hydra.yaml"
    local_metadata.parent.mkdir(parents=True)
    author_local_path = "/" + "data" + "/" + "wshf/private-run"
    local_metadata.write_text(f"cwd: {author_local_path}\n", encoding="utf-8")

    assert check_public_strings(tmp_path) == []


def test_public_string_scan_still_checks_release_files(tmp_path):
    public_document = tmp_path / "docs" / "installation.md"
    public_document.parent.mkdir(parents=True)
    author_local_path = "/" + "data" + "/" + "wshf/private-run"
    public_document.write_text(f"cwd: {author_local_path}\n", encoding="utf-8")

    assert check_public_strings(tmp_path) == [
        "docs/installation.md:1: author-local path"
    ]

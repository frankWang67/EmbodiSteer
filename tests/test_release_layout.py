from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_tree_is_self_contained_at_top_level():
    required = (
        "CITATION.cff",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "embodisteer",
        "diffusion_policy",
        "umi",
        "artifacts/manifest.yaml",
        "configs/policy/embodisteer.yaml",
        "third_party/manifest.yaml",
        "third_party/assets.yaml",
        "THIRD_PARTY_NOTICES.md",
        "docs/method.md",
        "eval_sim_single_robot.py",
        "eval_sim_multi_robots.py",
        "pyproject.toml",
        "environment/requirements.txt",
        "environment/dependencies.md",
    )
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"missing release paths: {missing}"

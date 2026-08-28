"""
Analyze a SLAM pipeline dataset to produce a summary report.
Usage: python scripts_slam_pipeline/analyze_dataset.py -i data/make_iced_coffee_ur5_100_0513
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import pathlib
import click
import pickle
import collections
import numpy as np


def analyze_slam_stdout(stdout_path, demo_name):
    """Extract key info from a slam_stdout.txt file."""
    if not stdout_path.is_file():
        return {"status": "no_stdout", "detail": "slam_stdout.txt not found"}

    text = stdout_path.read_text()

    if "CSV camera trajectory saved" in text:
        kf_match = [l for l in text.split("\n") if "Map 0 has" in l]
        kfs = kf_match[0].strip() if kf_match else "unknown KFs"
        status = "success"
    elif "Lost tracking on" in text:
        lost_lines = [l for l in text.split("\n") if "Lost tracking on" in l]
        lost_info = lost_lines[-1].strip() if lost_lines else "unknown reason"
        status = "lost_tracking"
    else:
        status = "other_failure"

    return {"status": status, "detail": "", "keyframes": None}


@click.command()
@click.option("-i", "--input", required=True, help="Path to dataset directory")
def main(input):
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath("demos")

    if not demos_dir.is_dir():
        print(f"ERROR: demos/ directory not found in {input_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Gather all demo dirs and check for key files
    # ------------------------------------------------------------------
    demo_dirs = sorted(demos_dir.glob("demo_*"))

    if len(demo_dirs) == 0:
        print("ERROR: No demo_* directories found!")
        sys.exit(1)

    with_csv = []
    without_csv = []
    with_tag = []
    without_tag = []
    slam_failures = {}  # demo_name -> failure type
    slam_kfs = {}  # demo_name -> keyframe count

    for d in demo_dirs:
        name = d.name
        has_csv = d.joinpath("camera_trajectory.csv").is_file()
        has_tag = d.joinpath("tag_detection.pkl").is_file()

        if has_csv:
            with_csv.append(name)
        else:
            without_csv.append(name)

        if has_tag:
            with_tag.append(name)
        else:
            without_tag.append(name)

        # Analyze SLAM stdout for failed or successful demos
        stdout_path = d.joinpath("slam_stdout.txt")
        if stdout_path.is_file():
            text = stdout_path.read_text()
            if "CSV camera trajectory saved" in text:
                kf_lines = [l.strip() for l in text.split("\n") if "Map 0 has" in l]
                kfs = kf_lines[0] if kf_lines else "?"
                slam_kfs[name] = kfs
            elif "Lost tracking on" in text:
                lost_lines = [l.strip() for l in text.split("\n") if "Lost tracking on" in l]
                slam_failures[name] = lost_lines[-1] if lost_lines else "lost_tracking"
            else:
                slam_failures[name] = "other"

    total_demos = len(demo_dirs)
    n_with_csv = len(with_csv)
    n_without_csv = len(without_csv)

    # ------------------------------------------------------------------
    # 2. Camera info from demo names
    # ------------------------------------------------------------------
    cam_serials = collections.Counter()
    for name in with_csv + without_csv:
        # demo_C3441326300000_2026.05.13_15.03.43.102267
        parts = name.split("_")
        if len(parts) >= 2:
            cam_serials[parts[1]] += 1

    # ------------------------------------------------------------------
    # 3. Check mapping and gripper_calibration
    # ------------------------------------------------------------------
    mapping_dir = demos_dir.joinpath("mapping")
    has_mapping = mapping_dir.is_dir()
    mapping_has_csv = mapping_dir.joinpath("camera_trajectory.csv").is_file() if has_mapping else False
    mapping_has_tag = mapping_dir.joinpath("tag_detection.pkl").is_file() if has_mapping else False
    has_tx_slam_tag = mapping_dir.joinpath("tx_slam_tag.json").is_file() if has_mapping else False

    gripper_dirs = sorted(demos_dir.glob("gripper_calibration_*"))
    has_gripper_cal = len(gripper_dirs) > 0

    # ------------------------------------------------------------------
    # 4. Read dataset_plan.pkl
    # ------------------------------------------------------------------
    plan_path = input_path.joinpath("dataset_plan.pkl")
    plan_info = None
    if plan_path.is_file():
        plan = pickle.load(open(plan_path, "rb"))
        total_episodes = len(plan)
        total_timesteps = sum(len(ep["episode_timestamps"]) for ep in plan)
        n_grippers = len(plan[0]["grippers"]) if plan else 0
        n_cameras = len(plan[0]["cameras"]) if plan else 0
        plan_info = {
            "episodes": total_episodes,
            "timesteps": total_timesteps,
            "grippers": n_grippers,
            "cameras": n_cameras,
        }

    # ------------------------------------------------------------------
    # 5. Compute "true" utilization vs reported
    # ------------------------------------------------------------------
    # Approximate: count all video files that generated a demo, regardless of SLAM
    video_files = sorted(input_path.glob("raw_videos/*.MP4"))
    # Exclude gripper_calibration videos
    raw_demo_videos = [v for v in video_files if "gripper_calibration" not in str(v.parent) and not v.parent.name.startswith("gripper")]

    # Count raw_videos in the raw_videos/ dir (excluding gripper_calibration subdir)
    raw_video_dir = input_path.joinpath("raw_videos")
    raw_mp4s = sorted([p for p in raw_video_dir.glob("*.MP4") if p.is_file()])

    # The total number of demos = total demo dirs
    # The "true" denominator = all demo dirs, while pipeline only counts those with csv

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    header = f" Dataset Analysis: {input_path.name} "
    print("=" * 70)
    print(f"{header:=^70}")
    print("=" * 70)

    # --- Section: Overview ---
    print(f"\n{'─' * 50}")
    print(" 1. DEMO OVERVIEW")
    print(f"{'─' * 50}")
    print(f"  Total demo directories:        {total_demos}")
    print(f"  With camera_trajectory.csv:     {n_with_csv}  ({n_with_csv/total_demos*100:.0f}%)")
    print(f"  Without camera_trajectory.csv:  {n_without_csv}  ({n_without_csv/total_demos*100:.0f}%)")
    if with_tag:
        print(f"  With tag_detection.pkl:         {len(with_tag)}")
    if without_tag:
        print(f"  Without tag_detection.pkl:      {len(without_tag)}")

    # --- Section: Cameras ---
    print(f"\n{'─' * 50}")
    print(" 2. CAMERAS")
    print(f"{'─' * 50}")
    for serial, count in cam_serials.most_common():
        print(f"  {serial}: {count} demos")

    # --- Section: SLAM failures ---
    if slam_failures:
        print(f"\n{'─' * 50}")
        print(" 3. SLAM FAILURES")
        print(f"{'─' * 50}")
        failure_types = collections.Counter()
        for name, detail in slam_failures.items():
            failure_types["Lost tracking (ORB-SLAM3 lost visual features)"] += 1
        for ftype, count in failure_types.most_common():
            print(f"  {ftype}: {count} demos")
        # Print the demos that failed
        if len(slam_failures) <= 20:
            print(f"\n  Failed demos:")
            for name in sorted(slam_failures.keys()):
                print(f"    {name}")
        else:
            print(f"\n  Failed demos (showing first 5 and last 5):")
            failed_names = sorted(slam_failures.keys())
            for name in failed_names[:5]:
                print(f"    {name}")
            print(f"    ... ({len(failed_names) - 10} more)")
            for name in failed_names[-5:]:
                print(f"    {name}")

    # --- Section: SLAM success (keyframes) ---
    if slam_kfs:
        print(f"\n{'─' * 50}")
        print(" 4. SLAM SUCCESS (Keyframes)")
        print(f"{'─' * 50}")
        print(f"  Successful demos: {len(slam_kfs)}")
        if len(slam_kfs) <= 10:
            for name, kfs in sorted(slam_kfs.items()):
                print(f"    {name}: {kfs}")
        else:
            kf_nums = []
            for name, kfs in sorted(slam_kfs.items()):
                # parse number of KFs from "Map 0 has XX KFs"
                try:
                    n = int(kfs.split("has ")[1].split(" KFs")[0])
                    kf_nums.append(n)
                except (ValueError, IndexError):
                    pass
            if kf_nums:
                print(f"  KF range: {min(kf_nums)} – {max(kf_nums)}, avg: {np.mean(kf_nums):.0f}")

    # --- Section: Missing camera_trajectory.csv ---
    if without_csv:
        print(f"\n{'─' * 50}")
        print(" 5. DEMOS WITHOUT camera_trajectory.csv")
        print(f"{'─' * 50}")
        if len(without_csv) <= 15:
            for name in without_csv:
                print(f"    {name}")
        else:
            print(f"  Count: {len(without_csv)}")
            print(f"  First 5:")
            for name in without_csv[:5]:
                print(f"    {name}")
            print(f"  Last 5:")
            for name in without_csv[-5:]:
                print(f"    {name}")

    # --- Section: Mapping ---
    print(f"\n{'─' * 50}")
    print(" 6. MAPPING & CALIBRATION")
    print(f"{'─' * 50}")
    print(f"  Mapping directory:          {'present' if has_mapping else 'MISSING'}")
    if has_mapping:
        print(f"    camera_trajectory.csv:    {'present' if mapping_has_csv else 'MISSING'}")
        print(f"    tag_detection.pkl:        {'present' if mapping_has_tag else 'MISSING'}")
        print(f"    tx_slam_tag.json:         {'present' if has_tx_slam_tag else 'MISSING'}")
    print(f"  Gripper calibration:        {'present' if has_gripper_cal else 'MISSING'}")
    if has_gripper_cal:
        for gd in gripper_dirs:
            has_range = gd.joinpath("gripper_range.json").is_file()
            print(f"    {gd.name}: gripper_range.json {'present' if has_range else 'MISSING'}")

    # --- Section: Dataset plan ---
    print(f"\n{'─' * 50}")
    print(" 7. DATASET PLAN (dataset_plan.pkl)")
    print(f"{'─' * 50}")
    if plan_info:
        print(f"  Total episodes:              {plan_info['episodes']}")
        print(f"  Total timesteps:             {plan_info['timesteps']}")
        print(f"  Grippers per episode:        {plan_info['grippers']}")
        print(f"  Cameras per episode:         {plan_info['cameras']}")
    else:
        print("  NOT FOUND - run 06_generate_dataset_plan.py first")

    # --- Section: Data utilization ---
    print(f"\n{'─' * 50}")
    print(" 8. DATA UTILIZATION")
    print(f"{'─' * 50}")
    print(f"  Demos that entered pipeline:  {n_with_csv} / {total_demos}")
    if plan_info:
        print(f"  Final episodes generated:     {plan_info['episodes']}")
        ratio = plan_info["episodes"] / total_demos * 100
        print(f"  Episodes / total demos:       {ratio:.0f}%")
    # Warning about misleading utilization
    if n_with_csv < total_demos:
        dropped = total_demos - n_with_csv
        print(f"\n  NOTE: {dropped} demos ({dropped/total_demos*100:.0f}% of total) were excluded from the")
        print(f"  pipeline's own utilization calculation because they lack")
        print(f"  camera_trajectory.csv. The pipeline-reported utilization only")
        print(f"  considers the {n_with_csv} demos that passed SLAM.")

    print(f"\n{'=' * 70}")
    print()


if __name__ == "__main__":
    main()

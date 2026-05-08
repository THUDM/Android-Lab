#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from pathlib import Path

import yaml


TASK_DIR_RE = re.compile(r"^(.+_\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def task_ids_from_configs(paths):
    task_ids = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = sorted(path.glob("*.yaml"))
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for task in data.get("tasks", []) or []:
                task_id = task.get("task_id")
                if task_id:
                    task_ids.append(str(task_id))
    return task_ids


def completed_task_ids(run_dir):
    completed = set()
    if not run_dir.exists():
        return completed
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = TASK_DIR_RE.match(child.name)
        if match:
            completed.add(match.group(1))
    return completed


def recent_lines(path, count):
    if not path or not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    interesting = [
        line for line in lines
        if (
            "Starting Android Emulator" in line
            or "Round " in line
            or "Completed successfully" in line
            or "Finished due to" in line
            or "Finished unexpectedly" in line
            or "Error:" in line
            or "Task " in line and "already run" in line
        )
    ]
    return interesting[-count:]


def finished_count(path):
    if not path or not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"Completed successfully\. \d+ rounds generated\.",
        r"Finished due to reaching max rounds\. \d+ rounds generated\.",
        r"Finished unexpectedly\. \d+ rounds generated\.",
    )
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def render_bar(done, total, width=30):
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_status(args):
    run_dir = Path(args.save_dir) / args.name
    task_ids = task_ids_from_configs(args.task_config)
    total = len(set(task_ids))
    started_ids = completed_task_ids(run_dir)
    started = len(started_ids)
    log_path = Path(args.log) if args.log else None
    finished = min(finished_count(log_path), total)
    pct = 0.0 if total == 0 else 100.0 * finished / total

    os.system("cls" if os.name == "nt" else "clear")
    print(f"Run: {args.name}")
    print(f"Directory: {run_dir}")
    print(f"Finished: {render_bar(finished, total)} {finished}/{total} ({pct:.1f}%)")
    print(f"Started task dirs: {started}/{total}")
    if total:
        remaining = sorted(set(task_ids) - started_ids)
        print(f"Remaining sample: {', '.join(remaining[:10]) if remaining else 'none'}")
    print()
    print("Recent log:")
    for line in recent_lines(log_path, args.lines):
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Monitor AndroidLab eval progress.")
    parser.add_argument("--name", required=True, help="Run name passed to eval.py -n")
    parser.add_argument("--save-dir", default="logs/evaluation")
    parser.add_argument("--task-config", nargs="+", default=["evaluation/config"])
    parser.add_argument("--log", default=None, help="Stdout/stderr log file for the run")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds")
    parser.add_argument("--lines", type=int, default=12)
    args = parser.parse_args()

    if args.watch and args.watch > 0:
        while True:
            print_status(args)
            time.sleep(args.watch)
    print_status(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate a video training manifest (JSONL) for video-language fine-tuning.

Each line must be a JSON object shaped like:

  {
    "video": "clip1.mp4",
    "messages": [
      {"role": "system", "content": "..."},        (optional)
      {"role": "user", "content": "Describe this clip."},
      {"role": "assistant", "content": "A person walks across a room."}
    ],
    "fps": 1.0,          (optional per-record override)
    "max_frames": 8      (optional per-record override)
  }

This performs REAL validation: it reads the manifest from disk, resolves
every "video" path against --videos-dir, and confirms the referenced file
actually exists and has a recognized video extension. The message structure
is validated with the same rules as the text dataset validator
(validate_dataset.py) since video records still need a real conversational
system/user/assistant shape for the model to learn from.

If the real `ffprobe` binary (bundled with ffmpeg) is on PATH, this also
probes each real video file for its real duration/resolution and reports
that in the errors list if the file cannot actually be decoded. If ffprobe
is not installed, video metadata probing is skipped and reported as an
honest warning rather than a failure — deep validation isn't required before
someone has installed ffmpeg, but a missing/invalid file always is.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_dataset import validate_record  # noqa: E402

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}


def probe_video(ffprobe_bin, video_path):
    """Runs the real ffprobe binary against a real file and returns (ok, info_or_error)."""
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()[:500] or "ffprobe could not read this file."
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, "ffprobe returned output that was not valid JSON."
    return True, parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to the manifest .jsonl file.")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Directory that contains the real video files referenced by the manifest. Defaults to the manifest's own folder.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(json.dumps({"ok": False, "error": f"File not found: {args.dataset}"}))
        sys.exit(1)

    videos_dir = args.videos_dir if args.videos_dir is not None else args.dataset.parent

    ffprobe_bin = shutil.which("ffprobe")
    probe_notes = []
    if ffprobe_bin is None:
        probe_notes.append(
            "ffprobe was not found on PATH, so real video metadata (duration/resolution) was not probed. "
            "File existence and JSON structure were still checked for real. Install ffmpeg to enable full probing."
        )

    total_lines = 0
    valid_lines = 0
    probed_files = 0
    all_errors = []

    with args.dataset.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            total_lines += 1

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                all_errors.append(f"Line {line_number}: invalid JSON ({exc.msg}).")
                continue

            if not isinstance(record, dict):
                all_errors.append(f"Line {line_number}: root value must be a JSON object.")
                continue

            line_errors = []

            video_name = record.get("video")
            if not isinstance(video_name, str) or not video_name.strip():
                line_errors.append(f"Line {line_number}: 'video' must be a non-empty string filename.")
            else:
                video_path = (videos_dir / video_name).resolve()
                extension = video_path.suffix.lower()
                if extension not in ALLOWED_VIDEO_EXTENSIONS:
                    line_errors.append(
                        f"Line {line_number}: '{video_name}' has an unrecognized video extension "
                        f"'{extension}'. Supported: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
                    )
                elif not video_path.is_file():
                    line_errors.append(f"Line {line_number}: referenced video file does not exist: {video_path}")
                elif ffprobe_bin:
                    probed_files += 1
                    ok, info = probe_video(ffprobe_bin, video_path)
                    if not ok:
                        line_errors.append(f"Line {line_number}: '{video_name}' failed a real ffprobe check: {info}")

            line_errors.extend(validate_record(record, line_number))

            if line_errors:
                all_errors.extend(line_errors)
            else:
                valid_lines += 1

    result = {
        "ok": len(all_errors) == 0,
        "file": str(args.dataset),
        "videos_dir": str(videos_dir),
        "total_records": total_lines,
        "valid_records": valid_lines,
        "invalid_records": total_lines - valid_lines,
        "videos_probed_with_ffprobe": probed_files,
        "notes": probe_notes,
        "errors": all_errors,
    }

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()

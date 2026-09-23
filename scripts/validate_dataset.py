#!/usr/bin/env python3
"""Validate a JSONL training dataset for QLoRA / TRL-style conversational fine-tuning.

This script performs REAL validation: it reads the file from disk, parses every
line as JSON, and reports the exact line number and problem for anything that is
wrong. It prints a machine-readable JSON summary to stdout so a caller (like the
Node backend in this repo) can show real, per-line results to the user.
"""

import argparse
import json
import sys
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


def validate_record(record, line_number):
    errors = []

    if not isinstance(record, dict):
        return [f"Line {line_number}: root value must be a JSON object."]

    messages = record.get("messages")

    if not isinstance(messages, list) or not messages:
        return [f"Line {line_number}: 'messages' must be a non-empty array."]

    roles = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"Line {line_number}, message {index}: message must be an object.")
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in VALID_ROLES:
            errors.append(f"Line {line_number}, message {index}: invalid role {role!r}.")
        else:
            roles.append(role)

        if not isinstance(content, str) or not content.strip():
            errors.append(f"Line {line_number}, message {index}: content must be a non-empty string.")

    if "user" not in roles:
        errors.append(f"Line {line_number}: no user message found.")

    if "assistant" not in roles:
        errors.append(f"Line {line_number}: no assistant message found.")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(json.dumps({"ok": False, "error": f"File not found: {args.dataset}"}))
        sys.exit(1)

    total_lines = 0
    valid_lines = 0
    all_errors = []

    # utf-8-sig transparently strips a UTF-8 byte-order-mark (BOM) if the file
    # has one, and behaves identically to plain utf-8 if it doesn't. Files
    # saved from Windows tools like Excel or Notepad often include a BOM,
    # which is still valid UTF-8 and should not be treated as an error.
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

            line_errors = validate_record(record, line_number)
            if line_errors:
                all_errors.extend(line_errors)
            else:
                valid_lines += 1

    result = {
        "ok": len(all_errors) == 0,
        "file": str(args.dataset),
        "total_records": total_lines,
        "valid_records": valid_lines,
        "invalid_records": total_lines - valid_lines,
        "errors": all_errors,
    }

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()

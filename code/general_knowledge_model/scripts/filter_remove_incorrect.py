#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input JSONL file")
    parser.add_argument("--backup-suffix", default=".bak", help="suffix for backup file")
    args = parser.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        print("input file not found:", inp, file=sys.stderr)
        sys.exit(2)

    out_dir = os.path.dirname(inp) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".tmp_filter_", text=True)
    removed = 0
    kept = 0

    with os.fdopen(tmp_fd, "w", encoding="utf-8") as out_f:
        with open(inp, "r", encoding="utf-8") as in_f:
            for line in in_f:
                s = line.rstrip("\n")
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    out_f.write(line)
                    kept += 1
                    continue

                if ("correct" in obj) and (obj.get("correct") is False):
                    removed += 1
                    continue

                out_f.write(line)
                kept += 1

    backup = inp + args.backup_suffix
    shutil.move(inp, backup)
    shutil.move(tmp_path, inp)

    print(f"removed={removed}")
    print(f"kept={kept}")
    print(f"backup={backup}")

if __name__ == '__main__':
    main()

"""Transfer all *_distilled.jsonl files from all/ to the selected/ directory."""
import shutil
from pathlib import Path

SRC = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/data/distilled/all")
DST = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/data/distilled/")

DST.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob("*_distilled.jsonl"))
print(f"Found {len(files)} distilled files to transfer")

for src_file in files:
    dst_file = DST / src_file.name
    shutil.copy2(src_file, dst_file)
    n = sum(1 for l in open(dst_file, encoding="utf-8") if l.strip())
    print(f"  {src_file.name} → {dst_file} ({n} examples)")

print(f"\nDone. {len(files)} files transferred to {DST}")
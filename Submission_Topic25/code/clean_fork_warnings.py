#!/usr/bin/env python3
"""
clean_fork_warnings.py — make the forked Ultralytics 8.0.201 run warning-free on
modern torch (2.8) / albumentations. Cosmetic only: these warnings never affected
results, but a clean log matters for the report and reproduction.

Run AFTER training finishes (it edits train.sh + fork training-loop files), then
verify with a 1-epoch RT-DETR smoke test showing zero warnings. Idempotent: each
patch is skipped if already applied; every target is asserted so a missed match
fails loudly instead of silently doing nothing.

Usage:  python clean_fork_warnings.py
"""
import re
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
FORK = PROJ / "diamond_detr" / "RTDETR-main"

# Safety: never edit train.sh / fork while a run is in progress.
if subprocess.run(["pgrep", "-f", "bash train.sh"], capture_output=True).returncode == 0:
    sys.exit("REFUSING: 'bash train.sh' is running. Run this only after training completes.")


def patch(path: Path, old: str, new: str, label: str):
    text = path.read_text()
    if new in text and old not in text:
        print(f"  [skip] {label} — already patched")
        return
    assert old in text, f"PATTERN NOT FOUND for {label} in {path} — fork changed?"
    path.write_text(text.replace(old, new))
    print(f"  [ok]   {label}")


print("Patching fork warnings...")

# 1. torch.cuda.amp.autocast / GradScaler -> torch.amp.* (modern API)
patch(FORK / "ultralytics/engine/trainer.py",
      "self.scaler = amp.GradScaler(enabled=self.amp)",
      "self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)",
      "trainer.py GradScaler")
patch(FORK / "ultralytics/engine/trainer.py",
      "with torch.cuda.amp.autocast(self.amp):",
      "with torch.amp.autocast('cuda', enabled=self.amp):",
      "trainer.py autocast")
# loss.py has 3 identical occurrences — patch() uses str.replace, hits all 3
patch(FORK / "ultralytics/utils/loss.py",
      "with torch.cuda.amp.autocast(enabled=False):",
      "with torch.amp.autocast('cuda', enabled=False):",
      "loss.py autocast x3")

# 2. albumentations: drop deprecated quality_lower arg (transform is p=0.0 anyway)
patch(FORK / "ultralytics/data/augment.py",
      "A.ImageCompression(quality_lower=75, p=0.0)]",
      "A.ImageCompression(p=0.0)]",
      "augment.py ImageCompression")

# 3. TensorBoard graph trace -> skip (source of the TB failure + TracerWarnings)
patch(FORK / "ultralytics/utils/callbacks/tensorboard.py",
      "WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])",
      "return  # graph logging disabled: noisy TracerWarnings on RT-DETR deformable attention",
      "tensorboard.py add_graph")

# 4. deterministic=False for the DETR runs (deformable attention has no deterministic
#    CUDA kernel; seed still controls init). Removes the grid_sampler warning.
ts = PROJ / "train.sh"
t = ts.read_text()
if "deterministic=False" not in t:
    t = t.replace('name="rtdetr_baseline", exist_ok=True,\n                                plots=False, $HP)',
                  'name="rtdetr_baseline", exist_ok=True,\n                                plots=False, deterministic=False, $HP)')
    t = t.replace('project="$CKPT", name="diamond_detr", exist_ok=True, plots=False,',
                  'project="$CKPT", name="diamond_detr", exist_ok=True, plots=False, deterministic=False,')
    ts.write_text(t)
    print("  [ok]   train.sh deterministic=False (both DETR models)")
else:
    print("  [skip] train.sh deterministic — already patched")

print("\nDone. Verify with a 1-epoch RT-DETR smoke test and confirm zero warnings.")

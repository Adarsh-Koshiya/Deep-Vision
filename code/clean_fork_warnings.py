#!/usr/bin/env python3
"""
Silence the warnings the forked Ultralytics 8.0.201 raises on modern torch (2.8) and
albumentations. The edits are cosmetic: they change no result, they only keep the
training log clean.

The script edits train.sh and a few fork files in place and is meant to run once
training has finished. It is idempotent (an applied patch is skipped on the next run),
and patch() asserts that each target string is present, so a pattern the fork has since
changed fails loudly rather than passing silently.

  python clean_fork_warnings.py
"""
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
FORK = PROJ / "diamond_detr" / "RTDETR-main"

# Guard: refuse to edit train.sh or the fork while a training run is in progress.
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
# loss.py has 3 identical occurrences; str.replace inside patch() catches all of them
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

# 4. deterministic=False for both DETR runs: deformable attention has no deterministic
#    CUDA kernel (seed still controls init). Silences the grid_sampler warning.
ts = PROJ / "train.sh"
t = ts.read_text()
if "deterministic=False" not in t:
    t = t.replace("plots=False, $HP_DETR)",                              # RT-DETR run
                  "plots=False, deterministic=False, $HP_DETR)")
    t = t.replace('name="diamond_detr", exist_ok=True, plots=False,',    # Diamond run
                  'name="diamond_detr", exist_ok=True, plots=False, deterministic=False,')
    ts.write_text(t)
    print("  [ok]   train.sh deterministic=False (both DETR models)")
else:
    print("  [skip] train.sh deterministic — already patched")

print("\nDone. Verify with a 1-epoch RT-DETR smoke test and confirm zero warnings.")

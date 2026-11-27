"""Regenerate all four Question 1 analysis sections."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("descriptive.py", "classification.py", "nonparametric.py", "evidence.py"):
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)

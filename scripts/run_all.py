"""
Runs every script in order, regenerating data/, figures/ and results/.
Takes about two minutes. Run from anywhere: python scripts/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

STEPS = [
    "table_C1_diffusivities",
    "fig2_sweep",
    "fig2_plot",
    "fig3_sweep",
    "fig3_plot",
    "c3_5_cross_flux_drag_check",
    "c4_isobaric_estimate",
    "c5_validation_tables",
    "length_scaling_check",
]

for name in STEPS:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(HERE / f"{name}.py")], capture_output=True, text=True)
    (RESULTS / f"{name}.log").write_text(proc.stdout + proc.stderr)
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"{name:<32}{status:<10}[{time.time() - t0:.0f} s]")
    if proc.returncode != 0:
        print(proc.stderr)
        sys.exit(proc.returncode)

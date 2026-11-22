"""Shared paths and helpers for the scripts (lets them run without installing the package)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
FIGURES = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

import numpy as np  # noqa: E402


def write_csv(path, columns):
    """columns: dict name -> 1-D array (all the same length)."""
    names = list(columns)
    arrays = [np.asarray(columns[n], dtype=float) for n in names]
    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for row in zip(*arrays):
            f.write(",".join(f"{v:.12e}" for v in row) + "\n")


def read_csv(path):
    with open(path) as f:
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {n: data[:, i] for i, n in enumerate(names)}

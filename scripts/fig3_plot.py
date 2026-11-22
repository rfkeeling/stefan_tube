"""
Fig. 3: O2 partial pressure at the end wall of the photosynthesis tube vs.
ambient humidity, full model (z_pivot/L = 0.5) and the closed-form Severinghaus
et al. prediction (Eq. 5), plus the numbers quoted in Sect. 5.2.

Right axis: pO2 / pO2_ref, with pO2_ref the closed-form value at zero humidity
gradient (chi_wall = chi_o, so pO2_ref = chi_o (1 - x_sat) p).

Reads data/fig3_photosynthesis_wall_o2.csv. Writes figures/fig3_pO2_vs_humidity.png
and data/fig3_pO2_vs_humidity.csv (the two plotted curves, in kPa).
"""
import _common
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stefan_tube import StefanTubeSolverPhotosynthesis, severinghaus_chi_wall

solver = StefanTubeSolverPhotosynthesis()
P, x_sat = solver.p, solver.x_sat

d = _common.read_csv(_common.DATA / "fig3_photosynthesis_wall_o2.csv")
x = d["x_H2O_ambient"]
pO2_full = d["x_O2_wall_pivot_0.50"] * P / 1e3  # kPa

chi_o = solver.chi_O2d_0
chi_wall = severinghaus_chi_wall(x, x_sat, chi_o)
pO2_sev = chi_wall * (1 - x_sat) * P / 1e3  # kPa

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(x*100, pO2_full, 'o-', color='#1f6f54', label='Full model', lw=2.5, ms=5, zorder=4)
ax.plot(x*100, pO2_sev, 's--', color='#888888', label='Severinghaus formula (closed form)',
        lw=2, ms=4.5, zorder=2)
ax.set_xlabel(r'$x_{H_2O}$ at $z=0$ (ambient)  (%)')
ax.set_ylabel(r'$pO_2$ at the wall  (kPa)')
ax.legend(fontsize=9.5)
both = np.concatenate([pO2_full, pO2_sev])
pad = 0.15 * (both.max() - both.min())
ax.set_ylim(both.min() - pad, both.max() + pad)

pO2_ref = chi_o * (1 - x_sat) * P / 1e3
ax2 = ax.twinx()
ax2.set_ylim(np.array(ax.get_ylim()) / pO2_ref)
ax2.set_ylabel(r'$pO_2 \,/\, pO_{2,ref}$')
ax2.ticklabel_format(axis='y', useOffset=False, style='plain')
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

plt.tight_layout()
plt.savefig(_common.FIGURES / "fig3_pO2_vs_humidity.png", dpi=150)
print("saved figures/fig3_pO2_vs_humidity.png")

_common.write_csv(_common.DATA / "fig3_pO2_vs_humidity.csv", {
    "x_H2O_ambient": x, "pO2_full_model_kPa": pO2_full, "pO2_Severinghaus_kPa": pO2_sev})
print("saved data/fig3_pO2_vs_humidity.csv\n")

print(f"Full model: pO2 at the wall falls {100*(pO2_full[0]-pO2_full[-1])/pO2_full[0]:.2f}% "
      f"as ambient x_H2O rises from {x[0]:.3f} to {x[-1]:.4f}")
print(f"Severinghaus closed form: falls {100*(pO2_sev[0]-pO2_sev[-1])/pO2_sev[0]:.2f}%")
print(f"Offset, full model minus closed form: {pO2_full[0]-pO2_sev[0]:.5f} kPa at zero humidity, "
      f"{pO2_full[-1]-pO2_sev[-1]:.5f} kPa at the humid end "
      f"(spread {np.ptp(pO2_full - pO2_sev)*1e3:.3f} Pa)")
spread = max(np.max(np.abs(d[f"x_O2_wall_pivot_{f}"] - d["x_O2_wall_pivot_0.50"])) for f in ("0.25", "0.75"))
print(f"Pivot independence: max |x_O2(wall; pivot 0.25 or 0.75) - x_O2(wall; pivot 0.50)| = {spread:.2e}")

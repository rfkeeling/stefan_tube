"""
Fig. 2: O2 flux vs. ambient humidity (from data/fig2_o2_flux_vs_humidity.csv),
plus the summary numbers quoted in Sect. 5.1 of the main text.

Writes figures/fig2_o2_flux_vs_humidity.png.
"""
import _common
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = _common.read_csv(_common.DATA / "fig2_o2_flux_vs_humidity.csv")
x = d["x_H2O_ambient"]
F10, F50, F100, F50_no = (d["F_O2_pivot_0.10"], d["F_O2_pivot_0.50"],
                          d["F_O2_pivot_1.00"], d["F_O2_pivot_0.50_no_cross_flux_drag"])

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(x*100, F10*1e3, 's--', color='#c96a2c', label=r'$z_{pivot}/L = 0.10$', lw=2, ms=4.5, zorder=2)
ax.plot(x*100, F50*1e3, 'o-', color='#1f6f54', label=r'$z_{pivot}/L = 0.50$', lw=2.5, ms=5, zorder=4)
ax.plot(x*100, F100*1e3, '^-.', color='#2f6f9e', label=r'$z_{pivot}/L = 1.00$', lw=2, ms=4.5, zorder=2)
ax.plot(x*100, F50_no*1e3, 'D:', color='#7a3b8f',
        label=r'$z_{pivot}/L = 0.50$, no cross-flux drag', lw=2, ms=4.5, zorder=3)
ax.set_xlabel(r'$x_{H_2O}$ at $z=0$ (ambient)  (%)')
ax.set_ylabel(r'O$_2$ flux, $F_{O_2}$  (mmol m$^{-2}$ s$^{-1}$)')
ax.set_title('O$_2$ flux vs. ambient humidity: pivot location and cross-flux drag')
ax.legend(fontsize=9.5)

# Right axis: flux relative to the base case at the humid end of the sweep.
F_ref = F50[-1] * 1e3
ax2 = ax.twinx()
ax2.set_ylim(np.array(ax.get_ylim()) / F_ref)
ax2.set_ylabel(r'$F_{O_2} \,/\, F_{O_2,ref}$')
ax2.ticklabel_format(axis='y', useOffset=False, style='plain')
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

plt.tight_layout()
plt.savefig(_common.FIGURES / "fig2_o2_flux_vs_humidity.png", dpi=150)
print("saved figures/fig2_o2_flux_vs_humidity.png\n")

print("Summary numbers (ambient x_H2O from 0 to x_sat = %.4f):" % x[-1])
for label, F in [("base case, z_pivot/L = 0.5", F50), ("z_pivot/L = 0.1", F10),
                 ("z_pivot/L = 1.0", F100), ("no cross-flux drag, z_pivot/L = 0.5", F50_no)]:
    print(f"  {label:<40} F_O2: {F[0]*1e3:.4f} -> {F[-1]*1e3:.4f} mmol m-2 s-1, "
          f"drop = {100*(F[0]-F[-1])/F[0]:.2f}%")
print(f"  simple Fick's-law scaling (proportional to x_O2): drop = {100*x[-1]:.2f}%")
print(f"  30% relative-humidity swing, scaled from the z_pivot/L = 1.0 sensitivity: "
      f"{0.30 * 100*(F100[0]-F100[-1])/F100[0]:.2f}%")

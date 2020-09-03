#!/usr/bin/env python3

from matplotlib import pyplot as plt
from PyPython import SpectrumUtils
from pathlib import Path


specs = []
for filename in Path(".").glob("**/*.log_spec_tot"):
    specs.append(str(filename))
print(specs)
nspec = len(specs)
fig, ax = plt.subplots(1, 1, figsize=(11.2, 6))
# ax = ax.flatten()
alpha = 0.8

l = ["-", "--"]

for i, file in enumerate(specs):
    print("plotting", file)
    s = SpectrumUtils.read_spec_file(file)
    ax.set_title(file)
    ax.loglog(s["Freq."].values, s["Freq."].values * s["Created"].values, label="Created: " + file, alpha=alpha, zorder=0)
    ax.loglog(s["Freq."].values, s["Freq."].values * s["Emitted"].values, label="Emitted: " + file, alpha=alpha)
    # ax.loglog(s["Freq."].values, s["Disk"].values, label="Disk", alpha=alpha)
    # ax.loglog(s["Freq."].values, s["Wind"].values, label="Wind", alpha=alpha)
    # ax.loglog(s["Freq."].values, s["Scattered"].values, label="Scattered", alpha=alpha)
    ax.legend()
    ax.set_xlabel("nu")
    ax.set_ylabel("nu Lnu")

ax.set_xlim(1e14, 1e18)
ax.set_ylim(1e24, 1e51)
SpectrumUtils.plot_line_ids(ax, SpectrumUtils.common_lines_list(freq=True), logx=True)

fig.tight_layout()
fig.savefig("spec_tot.pdf", dpi=300)
plt.show()

##
import numpy as np
import matplotlib.pyplot as plt

from pdos_surf.pdos_per_mode import *
from pdos_surf.momentum_relations import lambda_0
from pdos_surf.io_manager import *
from pdos_surf.util import *

base_dir='/Users/grunwal/Programming/pdos_surface_polaritons/data/'
WMAX=250.0
NW=20001
WLO=32.04
WTO=7.92

apdx = f'_wx{WMAX}_Nw{NW}'
wArr, zArr, pdos, params = load_bulk_pdos(WLO * 1e12, WTO * 1e12, base_dir=base_dir, apdx=apdx)
lambda0 = lambda_0(WLO * 1e12, WTO * 1e12, epsInf=1.0)

z_target = 0.1 * lambda0  # Set your target z value here
z_index = np.argmin(np.abs(zArr - z_target))
z_value = zArr[z_index]

# pdos shape: (n_modes, 2, len(zArr), len(wArr))
n_modes = pdos.shape[0]
mode_names = params['mode_names']
print(mode_names)

# Create figure with subplots
# %matplotlib qt
fig, ax = plt.subplots(figsize=(8 * CM, 6 * CM), dpi = 300)

pdos_full = np.sum(pdos[:, 0, z_index, :], axis = 0)

# Plot parallel component
for i in range(n_modes):
    pdos_para = pdos[i, 0, z_index, :]  # parallel component
    ax.plot(wArr / 1e12, pdos_para, label=mode_names[i])

ax.plot(wArr / 1e12, pdos_full, color = "black")

ax.set_xlim(0, 45)

ax.set_xlabel('Frequency ω (THz)')
ax.set_ylabel(r'$\rho_{\parallel}(\omega) / \rho_0(\omega)$')
ax.set_title(f'z = {z_value:.2e} m')
ax.axvline(WLO, color = "grey", lw = 0.9, ls = "--", zorder = 00, label=r'$\omega_LO$')
ax.axvline(WTO, color = "grey", lw = 0.9, ls = "--", zorder = 00, label=r'$\omega_TO$')
ax.legend()

plt.tight_layout()
# plt.savefig(base_dir + f'pdos_vs_frequency_z{z_index}{apdx}.png', dpi=300)
plt.show()

print(f"\nPlot saved to: {base_dir}pdos_vs_frequency_z{z_index}{apdx}.png")
print(f"z value: {z_value:.2e} m (index {z_index} out of {len(zArr)})")

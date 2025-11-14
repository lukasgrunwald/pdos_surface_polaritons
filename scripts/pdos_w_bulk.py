"""
Reproducing the frequency plot from Christians thesis
"""
##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pdos_surf.pdos_per_mode import *
from pdos_surf.momentum_relations import lambda_0, w_inf
from pdos_surf.io_manager import *
from pdos_surf.util import *
use_style('paper')

# —————————————————————————————————————————————————————————————————————————————————————— #
#                                       Bulk modes                                       #
# —————————————————————————————————————————————————————————————————————————————————————— #

base_dir='/Users/grunwal/Programming/pdos_surface_polaritons/data/'
WLO=32.04
WTO=7.92

# Parameters 1
# WMAX=250.0
# NW=20001
# ADD = ''

# Parameters 2
WMAX=250.0
NW=20001
ADD = '_l0'

'pdos_wLO32.04_wTO7.92_wx60.0_Nw5001_reference.h5'

apdx = f'_wx{WMAX}_Nw{NW}{ADD}'
wArr, zArr, pdos, params = load_bulk_pdos(WLO * 1e12, WTO * 1e12, base_dir=base_dir, apdx=apdx)
l0 = lambda_0(WLO * 1e12, WTO * 1e12, epsInf=1.0)
winf = w_inf(WLO * 1e12, WTO * 1e12, epsInf=1.0)
pdos *= 3/2 # Select one direction so that in w -> Inf it goes to one

# pdos shape: (n_modes, 2, len(zArr), len(wArr))
n_modes = pdos.shape[0]
mode_names = params['mode_names']

# Associated surface modes
w_surf = create_freq_array(w_bot=WTO * 1e12, w_top = winf, Nw=8001)
pdos_surf = pdos_para_perp_array(PdosSurf, w_surf, zArr, params['L'], WLO * 1e12, WTO * 1e12, 1., n_processes=1)

##- ———————————————————————————————————— Mode resolved ——————————————————————————————————— #
##
z_target = 10 * l0
z_index = np.argmin(np.abs(zArr - z_target))
z_value = zArr[z_index]

# Create figure with subplots
%matplotlib qt
fig, ax = plt.subplots(figsize=(8 * CM, 6 * CM), dpi = 300)
ax.set_prop_cycle('color', sns.color_palette('Set2'))

# Plot parallel component
for i in range(n_modes):
    label = mode_names[i][4:]
    pdos_para = pdos[i, 0, z_index, :]  # parallel component
    ax.plot(wArr / 1e12, pdos_para, label=f'${label}$')

ax.plot(wArr / 1e12, np.sum(pdos[:, 0, z_index, :], axis = 0), lw = 0.7, color = "black", label = r'$full$')
ax.axhline(1.0, zorder = 0, lw = 0.5, color = "grey", alpha = 0.5)

ax.set_xlim(0, 40)

ax.set_xticks([0., WTO, winf / 1e12, WLO])
ax.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)
# ax.set_yticks([0., 1, 2])
# ax.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

ax.set_xlabel(r'$\omega[THz]$')
ax.set_ylabel(r'$\rho_{x}(\omega) / \rho_{0, x}(\omega)$')
ax.set_title('$z =' + f'{z_value/l0:.2g}' + r'\lambda_0$', fontsize = 6)
ax.legend(ncol = 2, fontsize = 6)

plt.tight_layout()
plt.savefig(f'./figures/pdos_w/sto_bulk_pdos_resolved_{z_value/l0:.2f}.png', dpi=300)
plt.show()

##- —————————————————————————————————— end mode resolved ————————————————————————————————— #

##- —————————————————————————————————— Multiple z-values ————————————————————————————————— #
##
# z_tars = np.array([5, 1, 0.5, 0.3, 0.2, 0.1]) * l0 # Far-field
z_tars = np.array([0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02]) * l0 # Near-field
z_idxs = np.argmin(np.abs(zArr[:, None] - z_tars), axis=0)
z_vals = zArr[z_idxs]

pdos_bulk = np.sum(pdos[:, 0, z_idxs, :], axis = 0)

%matplotlib qt
fig, ax = plt.subplots(figsize=(8 * CM, 6 * CM), dpi = 300)
ax.set_prop_cycle('color', sns.color_palette('plasma', len(z_tars)))

# Plot parallel component
for i in range(len(z_tars)):
    # ax.plot(wArr / 1e12, pdos_bulk[i, :], label=r'$z/\lambda_\infty =' + f'{z_vals[i]/l0:1.0g}' + r' \lambda_\infty$')
    ax.plot(wArr / 1e12, pdos_bulk[i, :], lw = 0.7, label=r'$z/\lambda_\infty =' + f'{z_vals[i]/l0:1.0g}' + r'$')

ax.axhline(1.0, zorder = 0, lw = 0.5, color = "grey", alpha = 0.5)
ax.set_xlim(0, 40)
ax.set_ylim(0, )

ax.set_xticks([0., WTO, winf / 1e12, WLO])
ax.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)
# ax.set_yticks([0., 1, 2])
# ax.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

ax.set_xlabel(r'$\omega[THz]$')
ax.set_ylabel(r'$\rho_{x}(\omega) / \rho_{0, x}(\omega)$')
ax.set_title(r'$\textnormal{Bulk-modes}$', fontsize = 6)
ax.legend(ncol = 2, fontsize = 6)

plt.tight_layout()
if z_tars[-1]/l0 < 0.1:
    plt.savefig(f'./figures/pdos_w/sto_bulk_pdos_near_field.png', dpi=300)
else:
    plt.savefig(f'./figures/pdos_w/sto_bulk_pdos_far_field.png', dpi=300)
plt.show()

##- end multiple z-values

"""
Analysis for the frequency plot of surface modes
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

wLO = 32.04 * 1e12
wTO = 7.92 * 1e12
eps_inf = 1.0
L = 1.0

winf = w_inf(wLO, wTO, eps_inf)
l0 = lambda_0(wLO, wTO, eps_inf)

# Generate and safe the surface data
# w_vec = create_freq_array(w_bot=wTO, w_top = winf, Nw=20001)
# w_vec = safe_frequency_array(w_vec, wTO, wLO)
# z_vec = np.logspace(np.log10(1e2 * l0), np.log10(1e-5 * l0), 200, endpoint=True, base = 10)
# apdx = '_surface'
# pdos_surf = pdos_para_perp_array(PdosSurf, w_vec, z_vec, L, wLO, wTO, eps_inf, n_processes=1)
# save_bulk_pdos(pdos_surf[None, :], [PdosSurf, ], w_vec, z_vec, L, wLO, wTO, 1.0, base_dir='./data', overwrite=False, apdx=apdx)

w_vec, z_vec, pdos_surf, params = load_bulk_pdos(wLO, wTO, base_dir='./data', apdx='_surface')
pdos_ana = PdosSurfAnalytic.pdos_frequency_pos(z_vec, w_vec, wLO, wTO, eps_inf)

#! Strange correction that chrisitan implemented like this
epsilon = momentum_relations.epsilon(w_vec, wLO, wTO, eps_inf)
pdos_surf = pdos_surf * 1 / (1 + np.abs(epsilon)) * 3 / 2
pdos_ana = pdos_ana * 1 / (1 + np.abs(epsilon)) * 3 / 2

##- —————————————————————— Full PDOS compared to analytical resluts —————————————————————— #
##
z_tars = np.array([0.1, 0.01, 0.001, 0]) * l0
z_idxs = np.argmin(np.abs(z_vec[:, None] - z_tars), axis=0)
z_vals = z_vec[z_idxs]

%matplotlib qt
fig, ax = plt.subplots(figsize=(8 * CM, 6 * CM), dpi = 300)
ax.set_prop_cycle('color', sns.color_palette('plasma', len(z_tars)))
ax.set_yscale('log')

# Plot parallel component
for i in z_idxs:
    pdos_sum = np.sum(pdos_surf[0, :, i, :], axis = 0)
    # ax.plot(w_vec / 1e12, pdos_sum, marker = ".")
    # ax.plot(w_vec / 1e12, pdos_ana[i, :], color = 'black', ls = '--', lw = 0.7)
    ax.plot(w_vec / 1e12, np.abs(pdos_sum - pdos_ana[i, :]) / pdos_sum * 100)

ax.axhline(1.0, zorder = 0, lw = 0.5, color = "grey", alpha = 0.5)
ax.set_xlim(0, 40)
# ax.set_ylim(1e-1, )

ax.set_xticks([0., wTO / 1e12, winf / 1e12, wLO / 1e12])
ax.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$",
                     r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)
# ax.set_yticks([0., 1, 2])
# ax.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

ax.set_xlabel(r'$\omega[THz]$')
ax.set_ylabel(r'$\Delta \big(\rho_{x}(\omega) / \rho_{0, x}(\omega)\big)~[\%]$')
ax.set_title(r'$\textnormal{Surface-modes}$', fontsize = 6)
ax.legend(ncol = 2, fontsize = 6)

plt.tight_layout()
plt.savefig(f'./figures/pdos_w/sto_surface_total_pdos_error.png', dpi=300)
plt.show()


##- end comparison with analytical results

##- ————————————————————————— Parallel component as function of z ———————————————————————— #
##
z_tars = np.array([1, 0.1, 0.01, 0.001, 0]) * l0
# z_tars = np.array([5, 1, 0.5, 0.3, 0.2, 0.1]) * l0 # Far-field
z_idxs = np.argmin(np.abs(z_vec[:, None] - z_tars), axis=0)
z_vals = z_vec[z_idxs]

%matplotlib qt
fig, ax = plt.subplots(figsize=(8 * CM, 5.5 * CM), dpi = 300)
ax.set_prop_cycle('color', sns.color_palette('plasma', len(z_tars)))
ax.set_yscale('log')
ax.set_ylim(1e-6, 1e12)
ax.set_xlim(wTO / 1e12, wLO / 1e12)

# Plot parallel component
for i in z_idxs:
    label = r'$z/\lambda_\infty =' + f'{z_vec[i]/l0:1.0e}' + r'$'
    pdos_sum = np.sum(pdos_surf[0, :, i, :], axis=0)
    st, = ax.plot(w_vec / 1e12, pdos_surf[0, 0, i, :], lw=0.7, marker = ".", ms = 2, label=label, zorder = 10 - i)

# ax.set_xticks([0., wTO / 1e12, winf / 1e12, wLO / 1e12])
# ax.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)

ax.set_xticks([wTO / 1e12, winf / 1e12, wLO / 1e12])
ax.set_xticklabels([r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize = 8)
ax.annotate(r'$\epsilon(\omega) < 0$', xy = (0.05, 0.9), xycoords = 'axes fraction')

ax.set_ylabel(r'$\rho_{x}(\omega) / \rho_{0, x}(\omega)$')
ax.set_title(r'$\textnormal{Surface-modes}$', fontsize = 6)
ax.legend(ncol = 1, fontsize = 4)

plt.tight_layout()
if z_tars[-1]/l0 < 0.1:
    plt.savefig(f'./figures/pdos_w/sto_surface_para_pdos_near_field.png', dpi=300)
else:
    plt.savefig(f'./figures/pdos_w/sto_surface_para_pdos_far_field.png', dpi=300)
plt.show()

##- end parallel component

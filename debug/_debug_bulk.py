"""
Debugging calculation for parallel Bulk PDOS
"""
##
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import scipy.constants as consts
import matplotlib.pyplot as plt

from pdos_surf.pdos_per_mode import *
from pdos_surf.momentum_relations import lambda_0
from pdos_surf.io_manager import *
from pdos_surf.util import *

def load_dos_resolved(filepath):
    data = np.load(filepath)

    dosTE = data['dosTETotal']  # shape (3, n_freq, n_z)
    dosTM = data['dosTMPara']    # shape (3, n_freq, n_z)
    wArr = data['wArr']
    zArr = data['zArr']
    wLO = data['wLO'].item()
    wTO = data['wTO'].item()
    epsInf = data['epsInf'].item()

    return wArr, zArr, dosTE, dosTM, wLO, wTO, epsInf

filepath = '/Users/grunwal/Programming/pdos_surface_polaritons/data/reference_dos_resolved_wLO32.04_wTO7.92.npz'
base_dir = '/Users/grunwal/Programming/pdos_surface_polaritons/data'

# Load the original data
wArr, zArr, dosTE, dosTM, wLO, wTO, epsInf = load_dos_resolved(filepath)

'/Users/grunwal/Programming/pdos_surface_polaritons/data/pdos_wLO32.04_wTO7.92_wx60.0_Nw5001.h5'
'/Users/grunwal/Programming/pdos_surface_polaritons/data/pdos_wLO32.04_wTO7.92_wx60_Nw5001.h5'
wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
lambda0 = 2. * np.pi * consts.c / wInf
z_index = -1
z_value = zArr[z_index]

# Load new data
WMAX=40.0 # 60
NW=4001 # 5001
WLO=32.04
WTO=7.92

apdx = f'_wx{WMAX}_Nw{NW}_reference'
w_vec, z_vec, pdos, params = load_bulk_pdos(WLO * 1e12, WTO * 1e12, base_dir=base_dir, apdx=apdx)
mode_names = params['mode_names']
z_idx = np.argmin(np.abs(z_vec - z_value))

# Contribution labels
te_labels = ['TE', 'TE Eva', 'TE Res']
tm_labels = ['TM Para', 'TM Eva Para', 'TM Res Para']

%matplotlib qt
fig, ax = plt.subplots(ncols = 2, figsize=(20 * CM, 8 * CM))

# Plot TE contributions
for i in range(len(te_labels)):
    ax[0].plot(wArr / 1e12, dosTE[i, :, z_index], marker = ".", ms = 2, label=te_labels[i])
    ax[1].plot(wArr / 1e12, dosTM[i, :, z_index], marker = ".", ms = 2, label=tm_labels[i])

# TE Modes
ax[0].plot(w_vec / 1e12, pdos[0, 0, z_idx, :], marker = ".", ms = 2, ls = "--")
ax[0].plot(w_vec / 1e12, pdos[2, 0, z_idx, :], marker = ".", ms = 2, ls = "--")
ax[0].plot(w_vec / 1e12, pdos[4, 0, z_idx, :], marker = ".", ms = 2, ls = "--")
# ax.plot(wArr / 1e12, np.sum(dosTE[:, :, z_index], axis = 0), color = "black")

# TM Modes
ax[1].plot(w_vec / 1e12, pdos[1, 0, z_idx, :], ls = "--")
ax[1].plot(w_vec / 1e12, pdos[3, 0, z_idx, :], ls = "--")
ax[1].plot(w_vec / 1e12, pdos[5, 0, z_idx, :], ls = "--")

for it in ax:
    it.set_xlim(0, 60)
    it.axvline(wLO / 1e12, color='grey', ls='--', alpha=0.5, label='ω_LO')
    it.axvline(wTO / 1e12, color='grey', ls='--', alpha=0.5, label='ω_TO')
    it.set_xlabel('w')
    it.set_ylabel('TE DOS')
    it.set_title(f'z = {z_value:.2e} m')
    it.legend()
    it.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

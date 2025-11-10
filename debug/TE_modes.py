##
import numpy as np
import matplotlib.pyplot as plt

from pdos_surf.pdos_per_mode import *
from pdos_surf.io_manager import *

from debug.load_files import load_dos_resolved
# from debug.legacy.dosTEModes import calcDosTE
from debug.legacy.dosTEEvaModes import calcDosTE

base_dir = '/Users/grunwal/Programming/pdos_surface_polaritons/data'
filepath = base_dir + '/reference_dos_resolved_wLO32.04_wTO7.92.npz'

WMAX=40.0 # 60
NW=4001 # 5001
WLO=32.04
WTO=7.92
EPSINF = 2.0
L = 1.0 #! Not sure this is the correct thing to do!

# Load the reference data generated with christians code
w_ref, z_ref, pdosTE_ref, pdosTM_ref, wLO, wTO, epsInf = load_dos_resolved(filepath)
apdx = f'_wx{WMAX}_Nw{NW}_reference'
w_vec, z_vec, pdos, params = load_bulk_pdos(WLO * 1e12, WTO * 1e12, base_dir=base_dir, apdx=apdx)
dw = (w_vec[1] - w_vec[0]) / 1e12

z_idx_ref = -1
z_val = z_ref[z_idx_ref]
z_idx = np.argmin(np.abs(z_vec - z_val))

z_ref = np.array([1e-6, 1e-7, 1e-8, 1e-9])

# generate new array
# _wmin = np.argmin(np.abs(w_ref - WLO * 1e12)) - 1
# _wmax = np.argmin(np.abs(w_ref - (WLO + 10 * dw) * 1e12)) - 1
# w_new = w_ref[_wmin:_wmax]
w_new = np.linspace(5, WTO - 1e-2, 20) * 1e12

# On the fly calculation
pdos_new1 = pdos_para_perp_array(PdosTE, w_new, z_ref, L, wLO, wTO, epsInf, n_processes=None)

pdos_new = np.zeros((2, len(z_ref), len(w_new)))
pdos_leg = np.zeros((1, len(z_ref), len(w_new)))
for i in range(len(w_new)):
    para, perp = PdosEvaTE.pdos_para_perp_w(w_new[i], z_ref, L, wLO, wTO, epsInf)
    pdos_new[0, :, i] = para
    pdos_new[1, :, i] = perp

    pdos_leg[0, :, i] = calcDosTE(z_ref, L, w_new[i], wLO, wTO, epsInf)

##
%matplotlib qt
fig, ax = plt.subplots()

# ax.plot(w_ref[:] / 1e12, pdosTE_ref[1, :, -1], marker = "." , ms = 2)
# ax.plot(w_ref[_wmin:_wmax] / 1e12, pdosTE_ref[0, _wmin:_wmax, -1], marker = "." , ms = 2)
# ax.plot(w_vec / 1e12, pdos[0, 0, z_idx, :], marker = "." , ms = 2)

# apdx = f'_wx250.0_Nw20001'
# w_vec, z_vec, pdos, params = load_bulk_pdos(WLO * 1e12, WTO * 1e12, base_dir=base_dir, apdx=apdx)


z_idx_ref = -1
ax.plot(w_new / 1e12 , pdos_new[0, z_idx_ref, :], marker = "x", ms = 3, label = "direct")
ax.plot(w_new / 1e12, pdos_leg[0, z_idx_ref, :], marker = ".", ms = 2)

plt.legend()
plt.show()

"""
Tiny run script for generating and saving surface data. This is very quick and can in
principle also be done on the fly!
"""
##
import numpy as np

from pdos_surf.pdos_per_mode import *
from pdos_surf.momentum_relations import lambda_0, w_inf
from pdos_surf.io_manager import *
from pdos_surf.util import *

wLO = 32.04 * 1e12
wTO = 7.92 * 1e12
eps_inf = 1.0
L = 1.0

winf = w_inf(wLO, wTO, eps_inf)
l0 = lambda_0(wLO, wTO, eps_inf)

# Generate and safe the surface data
w_vec = create_freq_array(w_bot=wTO, w_top = winf, Nw=20001)
w_vec = safe_frequency_array(w_vec, wTO, wLO)
z_vec = np.logspace(np.log10(1e2 * l0), np.log10(1e-5 * l0), 200, endpoint=True, base = 10)

apdx = '_surface'
pdos_surf = pdos_para_perp_array(PdosSurf, w_vec, z_vec, L, wLO, wTO, eps_inf, n_processes=1)
save_bulk_pdos(pdos_surf[None, :], [PdosSurf, ], w_vec, z_vec, L, wLO, wTO, 1.0, base_dir='./data', overwrite=False, apdx=apdx)

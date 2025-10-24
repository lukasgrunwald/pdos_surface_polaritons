"""
Run file for generating bulk frequency resolved data
"""
##
import numpy as np
import scipy.constants as consts

from time import perf_counter
from tqdm import tqdm

import pdos_surf.io_manager as mg
import pdos_surf.momentum_relations as mr
from pdos_surf.pdos_per_mode import *
from pdos_surf.io_manager import *

def run_partition(n, N_partitions, apdx):
    """Run calculation for a single partition."""
    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    epsInf = 1.
    L = 1.0

    clsArr = [PdosTE, PdosTM, PdosEvaTE, PdosEvaTM, PdosResTE, PdosResTM]
    zArr = np.logspace(np.log10(1e-6), np.log10(1e-9), 200, endpoint=True, base=10)
    wArr = create_freq_array(w_top=60 * 1e12, Nw=5001)

    # Generate partitioned data
    base_dir = partition_folder_name(wLO, wTO, apdx)

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    tstart = perf_counter()
    generate_bulk_pdos_partition(n, N_partitions,
                                  clsArr, wArr, zArr,
                                  L, wLO, wTO, epsInf,
                                  n_processes=8, apdx=apdx)
    tend = perf_counter()
    print(f'Partition {n}/{N_partitions} took {tend - tstart:.2f} s')

if __name__ == '__main__':
    N_partitions = 100
    apdx = '_wx60_Nw5001'

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    epsInf = 1.
    L = 1.0

    clsArr = [PdosTE, PdosTM, PdosEvaTE, PdosEvaTM, PdosResTE, PdosResTM]
    zArr = np.logspace(np.log10(1e-6), np.log10(1e-9), 200,
                       endpoint=True, base=10)
    wArr = create_freq_array(w_top=60 * 1e12, Nw=5001)

    # Generate partitioned data
    w_part = partition_freq_array(wArr, N_partitions)
    base_dir = partition_folder_name(wLO, wTO, apdx)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for n in tqdm(np.arange(N_partitions)):
        tstart = perf_counter()
        generate_bulk_pdos_partition(n, N_partitions,
                                     clsArr, wArr, zArr,
                                     L, wLO, wTO, epsInf,
                                     n_processes=8, apdx=apdx)
        tend = perf_counter()
        print(f'Calculation took {tend - tstart} s')

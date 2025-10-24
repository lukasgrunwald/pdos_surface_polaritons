"""
Run file for generating bulk frequency resolved data
"""
import numpy as np
from time import perf_counter

from pdos_surf.pdos_per_mode import *
from pdos_surf.io_manager import *

def run_partition(n, N_partitions, wmax, Nw, wLO, wTO, apdx):
    """Run calculation for a single partition."""
    wLO = wLO * 1e12
    wTO = wTO * 1e12
    epsInf = 1.
    L = 1.0

    clsArr = [PdosTE, PdosTM, PdosEvaTE, PdosEvaTM, PdosResTE, PdosResTM]
    zArr = np.logspace(np.log10(1e-6), np.log10(1e-9), 200, endpoint=True, base=10)
    wArr = create_freq_array(w_top=wmax * 1e12, Nw=Nw)

    base_dir = partition_folder_name(wLO, wTO, apdx)
    os.makedirs(base_dir, exist_ok=True)

    tstart = perf_counter()
    generate_bulk_pdos_partition(n, N_partitions,
                                  clsArr, wArr, zArr,
                                  L, wLO, wTO, epsInf,
                                  n_processes=8, apdx=apdx)
    tend = perf_counter()
    print(f'Partition {n}/{N_partitions} took {tend - tstart:.2f} s')

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python script.py <n> <N_partitions> <wmax> <Nw> <wLO> <wTO>")
        sys.exit(1)

    print(sys.argv)
    n = int(sys.argv[1])
    N_partitions = int(sys.argv[2])
    wmax = float(sys.argv[3])  # In THz!
    Nw = int(sys.argv[4])
    wLO = float(sys.argv[5])  # In THz!
    wTO = float(sys.argv[6])  # In THz!
    apdx = f'_wx{wmax}_Nw{Nw}'

    run_partition(n, N_partitions, wmax, Nw, wLO, wTO, apdx)

import numpy as np
import scipy.constants as consts
from tqdm import tqdm

import pdos_surf.momentum_relations as mr
from pdos_surf.pdos_per_mode import *
from pdos_surf.io_manager import *


if __name__ == '__main__':
    N_partitions = 5
    apdx = '_test_run'

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    epsInf = 1.
    L = 1.0

    clsArr = [PdosTE, PdosTM, PdosEvaTE, PdosEvaTM, PdosResTE, PdosResTM]
    lambda0 = mr.lambda_0(wLO, wTO, epsInf)
    zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(5e-5 * lambda0), 200, endpoint=True, base=10)
    wArr = create_freq_array(w_top=50 * 1e12, Nw=31)

    # Generate partitioned data
    w_part = partition_freq_array(wArr, N_partitions)
    base_dir = partition_folder_name(wLO, wTO, apdx)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for n in tqdm(np.arange(N_partitions)):
        generate_bulk_pdos_partition(n, N_partitions,
                                     clsArr, wArr, zArr,
                                     L, wLO, wTO, epsInf,
                                     n_processes=8, apdx=apdx)

    merge_bulk_pdos_partitions(wLO, wTO, N_partitions, apdx=apdx)


# manual evaluation of each mode seperately
# pdos_TE = pdos_para_perp_array(PdosTE, wArr, zArr, L, wLO, wTO, epsInf, n_processes)
# pdos_EvaTE = pdos_para_perp_array(PdosEvaTE, wArr, zArr, L, wLO, wTO, epsInf, n_processes)
# pdos_ResTE = pdos_para_perp_array(PdosResTE, wArr, zArr, L, wLO, wTO, epsInf, n_processes)

# pdos_TM = pdos_para_perp_array(PdosTM, wArr, zArr, L, wLO, wTO, epsInf, n_processes)
# pdos_EvaTM = pdos_para_perp_array(PdosEvaTM, wArr, zArr, L, wLO, wTO, epsInf, n_processes)
# pdos_ResTM = pdos_para_perp_array(PdosResTM, wArr, zArr, L, wLO, wTO, epsInf, n_processes)
# tend = perf_counter()
# print(tend - tstart)

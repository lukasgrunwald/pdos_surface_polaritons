import numpy as np
from typing import Dict, Any

import os
import h5py

from .pdos_per_mode import *
from .util import *

# —————————————————————————————— Frequency array handling —————————————————————————————— #


def create_freq_array(*, w_top, Nw, w_bot=0.0, exclude_zero=True):
    wArr = np.linspace(w_bot, w_top, Nw)

    if exclude_zero and w_bot == 0:
        wArr = wArr[1:]

    return wArr


def safe_frequency_array(wArr: np.ndarray, wLO: float, wTO: float):
    """Remove potential divergencies in frequency array"""
    idx = np.where(np.abs(wArr - wTO) < 1e-6)

    if len(idx) > 0:
        dw = wArr[1] - wArr[0]
        wArr_safe = wArr.copy()
        wArr_safe[idx] += dw / 4
        return wArr_safe
    else:
        return wArr


def partition_freq_array(freq_array, n_subdivisions):
    array_length = len(freq_array)
    base_size = array_length // n_subdivisions
    remainder = array_length % n_subdivisions

    split_arrays = []
    start_idx = 0

    for i in range(n_subdivisions):
        # Add one extra element to first 'remainder' subdivisions
        size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + size
        split_arrays.append(freq_array[start_idx:end_idx])
        start_idx = end_idx

    return split_arrays


def merge_freq_array(split_arrays):
    return np.concatenate(split_arrays)


# ———————————————————————————————————— data handling ——————————————————————————————————— #


def pdos_filename(wLO, wTO, apdx=''):
    filename = f"pdos_wLO{wLO/1e12:.2f}_wTO{wTO/1e12:.2f}{apdx}.h5"

    return filename


def save_bulk_pdos(pdos, clsArr, wArr, zArr,
                   L, wLO, wTO, epsInf,
                   base_dir='./data',
                   overwrite=False,
                   apdx=''):
    filename = pdos_filename(wLO, wTO, apdx)
    path = os.path.join(base_dir, filename)
    path = versionized_path(path, overwrite)

    with h5py.File(path, 'w', track_order=True) as f:
        # Names of classes to be saved
        class_names = [cls.__name__ for cls in clsArr]

        # Save parameters as attributes
        f.attrs['wLO'] = wLO
        f.attrs['wTO'] = wTO
        f.attrs['epsInf'] = epsInf
        f.attrs['L'] = L
        f.attrs['mode_names'] = class_names

        # Iteration arrays
        f.create_dataset('wArr', data=wArr)
        f.create_dataset('zArr', data=zArr)

        # Save pdos in bare manner into file
        f.create_dataset('pdos', data=pdos)

    print(f"PDOS results saved to: {path}")
    return path


def generate_bulk_pdos(clsArr, wArr, zArr, L, wLO, wTO, epsInf,
                       n_processes=None,
                       base_dir='./data',
                       apdx=''):

    wArr = safe_frequency_array(wArr, wLO, wTO)
    pdos = pdos_para_perp_array_multi(clsArr, wArr, zArr, L, wLO, wTO, epsInf, n_processes)
    path = save_bulk_pdos(pdos, clsArr, wArr, zArr, L, wLO, wTO, epsInf,
                          base_dir=base_dir, apdx=apdx)

    return path


def _load_bulk_pdos(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load PDOS calculation results from HDF5 file.
    (n_modes, 2, len(zArr), len(wArr))

    Parameter dict contains: wLO, wTO, epsInf, L, mode_names
    """
    print(f'Reading file from {path}')

    with h5py.File(path, 'r') as f:
        # Load parameters from attributes
        params = dict(f.attrs)
        # Load arrays
        wArr = f['wArr'][:]
        zArr = f['zArr'][:]
        pdos = f['pdos'][:]

    return wArr, zArr, pdos, params


def load_bulk_pdos(wLO, wTO, base_dir='./data/', apdx=''):
    """
    returns  wArr, zArr, pdos [(n_modes, 2, len(zArr), len(wArr))], params
    """
    filename = pdos_filename(wLO, wTO, apdx)
    path = os.path.join(base_dir, filename)
    return _load_bulk_pdos(path)

# —————————————————————————————————— Workload_manager —————————————————————————————————— #


def generate_bulk_pdos_partition(n, N_partitions,
                                 clsArr, wArr, zArr, L, wLO, wTO, epsInf,
                                 n_processes=None,
                                 base_dir=None,
                                 apdx=''):

    if base_dir is None:
        base_dir = partition_folder_name(wLO, wTO, apdx)
    apdx = apdx + f'_P{n+1}-{N_partitions}'

    w_part = partition_freq_array(wArr, N_partitions)
    path = generate_bulk_pdos(clsArr, w_part[n], zArr, L, wLO, wTO, epsInf, n_processes,
                              base_dir=base_dir, apdx=apdx)
    return path


def partition_folder_name(wLO, wTO, apdx=''):
    filename = pdos_filename(wLO, wTO, apdx)
    base, _ = os.path.splitext(filename)
    base_dir = f'./data/tmp_part_{base}'
    return base_dir


def merge_bulk_pdos_partitions(wLO, wTO, N_partitions,
                               base_dir=None,
                               output_dir='./data',
                               apdx='',
                               overwrite=False):
    """ Merge partitioned PDOS files into a single file. """
    if base_dir is None:
        base_dir = partition_folder_name(wLO, wTO, apdx)

    # Load all partitions
    wArr_vec = []
    pdos_vec = []
    params = None
    zArr = None

    for n in range(N_partitions):
        part_apdx = apdx + f'_P{n+1}-{N_partitions}'
        filename = pdos_filename(wLO, wTO, part_apdx)
        path = os.path.join(base_dir, filename)
        print(path)

        wArr_part, zArr_part, pdos_part, params_part = _load_bulk_pdos(path)

        wArr_vec.append(wArr_part)
        pdos_vec.append(pdos_part)

        # Store params and zArr from first partition
        if params is None:
            params = params_part
            zArr = zArr_part

    # Concatenate frequency arrays and pdos along frequency axis (axis=-1)
    wArr_merged = np.concatenate(wArr_vec)
    pdos_merged = np.concatenate(pdos_vec, axis=-1)

    # Get class array from params
    clsArr = [eval(name) for name in params['mode_names']]
    # Save merged file
    path = save_bulk_pdos(pdos_merged, clsArr, wArr_merged, zArr,
                          params['L'], wLO, wTO, params['epsInf'],
                          base_dir=output_dir,
                          overwrite=overwrite,
                          apdx=apdx)

    print(f"Merged {N_partitions} partitions into: {path}")
    return path

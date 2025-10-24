from pdos_surf.pdos_per_mode import *
from pdos_surf.io_manager import *

N_PARTITIONS=10
WMAX=60.0
NW=101
WLO=32.04
WTO=7.92

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

    merge_bulk_pdos_partitions(wLO, wTO, N_partitions, apdx=apdx)

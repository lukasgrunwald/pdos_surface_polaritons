from pdos_surf.pdos_per_mode import *
from pdos_surf.io_manager import *

N_PARTITIONS=5
WMAX=60.0
NW=101
WLO=32.04
WTO=7.92

apdx = f'_wx{WMAX}_Nw{NW}'
merge_bulk_pdos_partitions(WLO * 1e12, WTO * 1e12, N_PARTITIONS, apdx=apdx)

#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=sto_pdos
#SBATCH --output=cluster/logs/%x_%A_%a.out

#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=01:00:00
#SBATCH --account=theophysc

# Navigate to project folder and setup logs
date
cd $HOME/Dokumente/pdos_surface_polaritons
mkdir -p cluster/logs

module load Python/3.12.3
source $HOME/venv/Default/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Set parameters (all frequencies in THz)
N_PARTITIONS=${SLURM_ARRAY_TASK_COUNT}
WMAX=250.0
NW=20001
WLO=32.04
WTO=7.92

# Run the Python script with the array task ID as partition index
python cluster/pdos_partition_run.py ${SLURM_ARRAY_TASK_ID} ${N_PARTITIONS} ${WMAX} ${NW} ${WLO} ${WTO}
echo "Completed partition $((SLURM_ARRAY_TASK_ID + 1))/${N_PARTITIONS}"

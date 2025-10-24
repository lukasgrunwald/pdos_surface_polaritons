#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=surface_pdos
#SBATCH --output=logs/pdos_%A_%a.out
#SBATCH --error=logs/pdos_%A_%a.err

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=06:00:00
#SBATCH --account=theophysc
###SBATCH --partition=YOUR_PARTITION
###SBATCH --array=0-99

#! Need to update this!
### Output files for the program (%x is the job-name and %j the job id)
#SBATCH -o /home/yn858207/Dokumente/syk/cluster/outputs/O_%x_%j.txt
#SBATCH -e /home/yn858207/Dokumente/syk/cluster/outputs/E_%x_%j.txt

# Setup output folder

# Load relevant python modules
module load python/3.x
module load scipy-stack  # or whatever provides numpy/scipy
# source /path/to/your/venv/bin/activate

# Navigate to folder and activate environment

### Set parameters (all frequencies in THz)
N_PARTITIONS=100
WMAX=60.0
NW=101
WLO=32.04
WTO=7.92

### Run the Python script with the array task ID as partition index
### python your_script.py ${SLURM_ARRAY_TASK_ID} ${N_PARTITIONS} ${WMAX} ${NW} ${WLO} ${WTO}
python your_script.py 0 ${N_PARTITIONS} ${WMAX} ${NW} ${WLO} ${WTO}

echo "Completed partition ${SLURM_ARRAY_TASK_ID}/${N_PARTITIONS}"

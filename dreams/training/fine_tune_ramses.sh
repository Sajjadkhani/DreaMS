#!/bin/bash -l
#SBATCH --job-name=DreaMS_fine-tuning
#SBATCH --account=ag-brodesser
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH -G h100:2
#SBATCH --time 72:00:00
#SBATCH --mem=128gb
#SBATCH --output=DreaMS_fine-tuning_%j.out
#SBATCH --error=DreaMS_fine-tuning_%j.err
#SBATCH --mail-user=skhani1@uni-koeln.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# Load necessary module
module purge
module load lang/Miniconda3/23.9.0-0
module load /projects/ramses-software/modules/accel/cuda/12.5

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dreams

# Export project definitions
$(python -c "from dreams.definitions import export; export()")

# Move to running dir
cd "${DREAMS_DIR}" || exit 3

# Run the training script
# Replace `python3 training/train.py` with `srun --export=ALL --preserve-env python3 training/train.py \`
# when executing on a SLURM cluster via `sbatch`.
srun --export=ALL --preserve-env python3 training/train.py \
  --project_name MolecularProperties \
  --job_key "my_run_name" \
  --run_name "my_run_name" \
  --train_objective mol_props \
  --train_regime fine-tuning \
  --dataset_pth "${DATA_DIR}/GeMS/auxiliary/MassSpecGym_MurckoHist_split.hdf5" \
  --dformat A \
  --model DreaMS \
  --lr 3e-5 \
  --batch_size 64 \
  --prec_intens 1.1 \
  --num_devices 2 \
  --max_epochs 103 \
  --log_every_n_steps 5 \
  --head_depth 1 \
  --seed 3407 \
  --train_precision 64 \
  --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
  --val_check_interval 0.1 \
  --max_peaks_n 100 \
  --save_top_k -1 \
  --no_wandb \
  --num_workers_data 64

#!/bin/bash -l
#SBATCH --job-name=DreaMS_pre-training
#SBATCH --account=ag-brodesser
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH -G h100:2
#SBATCH --time 72:00:00
#SBATCH --mem=128gb
#SBATCH --output=DreaMS_pre-training_%j.out
#SBATCH --error=DreaMS_pre-training_%j.err
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

job_key="my_pre_training_run"

# Run the training script
# Replace `python3 training/train.py` with `srun --export=ALL --preserve-env python3 training/train.py \`

# when executing on a SLURM cluster via `sbatch`.
srun --export=ALL --preserve-env python3 training/train.py \
 --project_name SSL_VAL_4.0 \
 --job_key "${job_key}" \
 --run_name "${job_key}" \
 --frac_masks 0.3 \
 --train_regime pre-training \
 --dataset_pth "${GEMS_DIR}/GeMS_A/GeMS_A10.hdf5" \
 --val_check_interval 0.1 \
 --train_objective mask_mz_hot \
 --hot_mz_bin_size 0.05 \
 --dformat A \
 --model DreaMS \
 --ff_peak_depth 1 \
 --ff_fourier_depth 5 \
 --ff_fourier_d 512 \
 --ff_out_depth 1 \
 --prec_intens 1.1 \
 --num_devices 2 \
 --max_epochs 3000 \
 --log_every_n_steps 20 \
 --seed 3402 \
 --n_layers 7 \
 --n_heads 8 \
 --d_peak 44 \
 --d_fourier 980 \
 --lr 1e-4 \
 --batch_size 256 \
 --dropout 0.1 \
 --save_top_k -1 \
 --att_dropout 0.1 \
 --residual_dropout 0.1 \
 --ff_dropout 0.1 \
 --weight_decay 0 \
 --attn_mech dot-product \
 --train_precision 32 \
 --mask_peaks \
 --mask_intens_strategy intens_p \
 --max_peaks_n 60 \
 --ssl_probing_depth 0 \
 --focal_loss_gamma 5 \
 --no_transformer_bias \
 --n_warmup_steps 5000 \
 --fourier_strategy lin_float_int \
 --mz_shift_aug_p 0.2 \
 --mz_shift_aug_max 50 \
 --pre_norm \
 --graphormer_mz_diffs \
 --ret_order_loss_w 0.2 \
 --no_wandb \
 --num_workers_data 64 # the max--num_workers 96



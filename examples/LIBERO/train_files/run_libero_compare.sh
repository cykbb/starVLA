#!/bin/bash
#SBATCH -J qwenpi_libero_cmp
#SBATCH -p h200n
#SBATCH -A prj0000000267
#SBATCH -t 0-18
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

############################
# Environment
############################
module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1
eval "$(conda shell.bash hook)"
conda activate /scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export OMP_NUM_THREADS=1

############################
# Paths & config
############################
cd /home/users/astar/i2r/lishijie/yk/starVLA || exit 1
echo "Current directory: $(pwd)"

Framework_name=QwenPI
freeze_module_list='visual'
base_vlm=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_libero_vla_only.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=/home/users/astar/i2r/lishijie/grasping_challenge/scratch/results/Checkpoints
run_id=${RUN_ID:-qwenpi_libero_vla_only_cmp}  # allow override via env

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
ln -sfn "${run_root_dir}" "$(pwd)/results/Checkpoints"
cp "$0" "${output_dir}/" || true

############################
# Launch
############################
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 32 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 1000 \
  --trainer.eval_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_entity bykkk-nanyang-technological-university-singapore \
  --wandb_project starVLA_Libero

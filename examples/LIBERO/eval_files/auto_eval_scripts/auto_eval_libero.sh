#!/bin/bash
#SBATCH -J autoeval_libero
#SBATCH -p h200n
#SBATCH -A prj0000000267
#SBATCH -t 0-20
#SBATCH -N 1
#SBATCH --ntasks-per-node=8          # 跑四个基准各占一张卡，余下可留给系统/冗余
#SBATCH --gres=gpu:8
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

# 项目根目录
cd /home/users/astar/i2r/lishijie/yk/starVLA

# 主脚本路径
SCRIPT_PATH="./examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"

# 默认权重（可按需修改）
your_ckpt=/home/users/astar/i2r/lishijie/grasping_challenge/scratch/results/Checkpoints/padtpi_libero_vla_only/checkpoints/steps_5000_pytorch_model.pt

# run_index 基准，用于分配 GPU 与端口（6450+run_index）
run_index_base=0

#####################################################
task_suite_name=libero_10 # align with your model
run_index=$((run_index_base + 0))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
#####################################################
task_suite_name=libero_object # align with your model
run_index=$((run_index_base + 2))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
####################################################
task_suite_name=libero_spatial # align with your model
run_index=$((run_index_base + 3))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

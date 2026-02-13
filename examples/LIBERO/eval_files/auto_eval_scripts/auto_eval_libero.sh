#!/bin/bash
#SBATCH -J autoeval_libero
#SBATCH -p h200n            # 队列/partition
#SBATCH -A prj0000000267          # 项目号/Account
#SBATCH -t 0-10                   # 运行时间：0-10 = 10小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --ntasks-per-node=8       # 每节点 8 个任务
#SBATCH --gres=gpu:8              # 申请 8 张 GPU
#SBATCH -o slurm_%x_%j.out        # 标准输出
#SBATCH -e slurm_%x_%j.err        # 错误输出

set -euo pipefail

# 项目根目录
cd /home/users/astar/i2r/lishijie/yk/starVLA

# 主脚本路径
SCRIPT_PATH="./examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"

# 默认权重（可按需修改）
#your_ckpt=/home/users/astar/i2r/lishijie/grasping_challenge/scratch/results/Checkpoints/padtpi_libero_pw_with_grads/checkpoints/steps_15000_pytorch_model.pt
your_ckpt=/home/users/astar/i2r/lishijie/grasping_challenge/scratch/results/Checkpoints/padtpi_libero_vla_pw_only/checkpoints/steps_15000_pytorch_model.pt
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

wait

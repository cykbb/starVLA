#!/bin/bash
#SBATCH -J qwenpi_libero
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A prj0000000267          # 项目号/Account
#SBATCH -t 0-12                   # 运行时间：0-12 = 12小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --ntasks-per-node=1       # 每节点 4 个任务
#SBATCH --gres=gpu:1              # 申请 4 张 GPU
#SBATCH -o slurm_%x_%j.out        # 标准输出
#SBATCH -e slurm_%x_%j.err        # 错误输出

set -euo pipefail

############################
# 加载环境（按图示例）
############################
# 完全清理所有模块，避免冲突
module purge

module load miniforge/24.11.3-2
module load cuda/12.4.1

# 初始化 conda（必须在 SLURM 脚本中）
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate /scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3


export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# 保留旧变量（兼容一些老脚本/组件）
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export OMP_NUM_THREADS=1

############################
# 切换到项目根目录
############################
cd /home/users/astar/i2r/lishijie/yk/starVLA || exit 1
echo "Current directory: $(pwd)"

python test.py
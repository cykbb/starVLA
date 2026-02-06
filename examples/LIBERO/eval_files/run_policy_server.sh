#!/bin/bash
#SBATCH -J policy_server
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A prj0000000267          # 项目号/Account
#SBATCH -t 0-12                   # 运行时间：0-12 = 12小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --ntasks-per-node=1       # 每节点 1 个任务
#SBATCH --gres=gpu:1              # 申请 1 张 GPU
#SBATCH -o slurm_%x_%j.out        # 标准输出
#SBATCH -e slurm_%x_%j.err        # 错误输出

set -euo pipefail

############################
# 加载环境
############################
module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1

# 初始化 conda
eval "$(conda shell.bash hook)"

# 切换到项目根目录
cd /home/users/astar/i2r/lishijie/yk/starVLA || exit 1
echo "Current directory: $(pwd)"

# 激活 conda 环境
conda activate /scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA

export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

your_ckpt=results/Checkpoints/qwenpi_libero_vla_only/checkpoints/steps_15000_pytorch_model.pt
port=5694

################# star Policy Server ######################
echo "Starting policy server with checkpoint: ${your_ckpt}"
echo "Listening on port: ${port}"

# export DEBUG=true
python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

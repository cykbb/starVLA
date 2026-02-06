#!/bin/bash
#SBATCH -J padtpi_libero
#SBATCH -p testqueue              # 队列/partition
#SBATCH -A prj0000000267          # 项目号/Account
#SBATCH -t 0-12                   # 运行时间：0-12 = 12小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --ntasks-per-node=2       # 每节点 2 个任务
#SBATCH --gres=gpu:2              # 申请 2 张 GPU
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

Framework_name=PaDTPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_libero_padt_vla_only.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=padtpi_libero_vla_only



output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/" || true

############################
# 启动训练
############################
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_entity bykkk-nanyang-technological-university-singapore \
  --wandb_project starVLA_Libero


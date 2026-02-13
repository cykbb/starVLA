#!/bin/bash
#SBATCH -J padtpi_libero
#SBATCH -p testqueue            # 队列/partition
#SBATCH -A prj0000000267          # 项目号/Account
#SBATCH -t 0-20                   # 运行时间：0-20 = 20小时
#SBATCH -N 1                      # 1 个节点
#SBATCH --ntasks-per-node=2       # 每节点 2 个任务
#SBATCH --gres=gpu:2              # 申请 2 张 GPU
#SBATCH -o slurm_%x_%j.out        # 标准输出
#SBATCH -e slurm_%x_%j.err        # 错误输出
set -euo pipefail

############################
# 加载环境（按图示例）
############################
module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1

# 初始化 conda（必须在 SLURM 脚本中）
eval "$(conda shell.bash hook)"

cd "/home/users/astar/i2r/lishijie/yk/starVLA"
conda activate /scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA


###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME="/home/users/astar/i2r/lishijie/yk/LIBERO"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"


export LIBERO_Python="/scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA/bin/python"

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${LIBERO_HOME}"   # 追加 LIBERO_HOME（无多余冒号）
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"          # 把 repo 根目录放前面

CONFIG_FILE="${LIBERO_CONFIG_PATH%/}/config.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${CONFIG_FILE}" <<EOF
benchmark_root: "${LIBERO_HOME}/libero"
bddl_files: "${LIBERO_HOME}/libero/bddl_files"
init_states: "${LIBERO_HOME}/libero/init_files"
datasets: "/home/users/astar/i2r/lishijie/yk/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA"
assets: "${LIBERO_HOME}/libero/assets"
EOF
fi

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt="/home/users/astar/i2r/lishijie/grasping_challenge/scratch/results/Checkpoints/padtpi_libero_vla_only/checkpoints/steps_5000_pytorch_model.pt"
export DEBUG=true

folder_name="$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

task_suite_name="libero_goal"
num_trials_per_task=5  # 减少到5次试验进行快速测试
video_out_path="results/${task_suite_name}/${folder_name}"

export SAVE_VIDEO=True

"${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${your_ckpt}" \
  --args.host "${host}" \
  --args.port "${base_port}" \
  --args.task-suite-name "${task_suite_name}" \
  --args.num-trials-per-task "${num_trials_per_task}" \
  --args.video-out-path "${video_out_path}"

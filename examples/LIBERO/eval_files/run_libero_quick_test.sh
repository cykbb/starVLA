#!/bin/bash
#SBATCH -J libero_goal_quick
#SBATCH -p testqueue              
#SBATCH -A prj0000000267          
#SBATCH -t 0-6                    # 6小时运行时间
#SBATCH -N 1                      
#SBATCH --ntasks-per-node=1       
#SBATCH --gres=gpu:2              
#SBATCH -o slurm_%x_%j.out        
#SBATCH -e slurm_%x_%j.err        

set -euo pipefail

############################
# 加载环境
############################
module purge
module load miniforge/24.11.3-2
module load cuda/12.4.1

eval "$(conda shell.bash hook)"

cd "/home/users/astar/i2r/lishijie/yk/starVLA"
conda activate /scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA

# 设置环境变量
export LIBERO_HOME="/home/users/astar/i2r/lishijie/yk/LIBERO"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export LIBERO_Python="/scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA/bin/python"

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${LIBERO_HOME}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# 创建配置文件
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

# 设置参数
host="127.0.0.1"
port=5694
your_ckpt="/home/users/astar/i2r/lishijie/yk/starVLA/results/Checkpoints/qwenpi_libero_vla_only/checkpoints/steps_15000_pytorch_model.pt"
export DEBUG=true

folder_name="$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

task_suite_name="libero_goal"
num_trials_per_task=10  # 快速测试：每个任务10次试验
video_out_path="results/${task_suite_name}/${folder_name}"

echo "Starting policy server with checkpoint: ${your_ckpt}"
echo "Listening on port: ${port}"

# 在后台启动策略服务器
python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 &

SERVER_PID=$!
echo "Policy server started with PID: ${SERVER_PID}"

# 等待服务器启动
sleep 15

# 启动评估
echo "Starting evaluation with ${num_trials_per_task} trials per task..."
"${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${your_ckpt}" \
  --args.host "${host}" \
  --args.port "${port}" \
  --args.task-suite-name "${task_suite_name}" \
  --args.num-trials-per-task "${num_trials_per_task}" \
  --args.video-out-path "${video_out_path}"

# 清理后台进程
echo "Evaluation completed. Stopping policy server..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true
echo "Policy server stopped."
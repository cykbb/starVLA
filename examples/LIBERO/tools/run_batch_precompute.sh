#!/bin/bash
#SBATCH -J libero_batch
#SBATCH -p testqueue
#SBATCH -A prj0000000267
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:3          # 使用3个GPU
#SBATCH -o batch_precompute.out
#SBATCH -e batch_precompute.err

# 1. 进入代码目录
cd "/home/users/astar/i2r/lishijie/yk/starVLA"

# 2. 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/prj0000000267/grasping_challenge/.conda/envs/starVLA

# 3. 【核心】解决报错必须加这两句
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. 设置LIBERO路径
export LIBERO_HOME="${LIBERO_HOME:-/home/users/astar/i2r/lishijie/yk/LIBERO}"
export PYTHONPATH="${LIBERO_HOME}:${PYTHONPATH:-}"

echo "=========================================="
echo "开始批量处理LIBERO数据集"
echo "时间: $(date)"
echo "LIBERO_HOME: $LIBERO_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python版本: $(python --version)"
echo "当前目录: $(pwd)"
echo "可用GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 批量处理四个任务套件
# 使用max_workers=1避免多进程GPU冲突（即使有2个GPU，单进程顺序处理更稳定）
python -u examples/LIBERO/tools/batch_precompute_libero.py \
  --base_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --task_suites libero_spatial libero_goal libero_object libero_10 \
  --max_workers 1 \
  --resolution 256 \
  --seed 0 \
  --mappings_output examples/LIBERO/tools/libero_id_to_label_mappings.json 2>&1

EXIT_CODE=$?
echo "=========================================="
echo "批量处理完成"
echo "退出码: $EXIT_CODE"
echo "时间: $(date)"
echo "=========================================="

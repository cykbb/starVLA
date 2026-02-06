#!/bin/bash
#SBATCH -J libero_sim
#SBATCH -p testqueue
#SBATCH -A prj0000000267
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1          # 必须要有GPU
#SBATCH -o sim1.out            # 日志直接输出到当前目录
#SBATCH -e sim1.err

# 1. 进入代码目录
cd "/home/users/astar/i2r/lishijie/yk/starVLA"

# 2. 激活环境 (直接用你的环境路径)
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
echo "开始运行预计算脚本"
echo "时间: $(date)"
echo "LIBERO_HOME: $LIBERO_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python版本: $(python --version)"
echo "当前目录: $(pwd)"
echo "=========================================="
python examples/LIBERO/tools/extract_episode_images.py \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000.parquet \
    --output_dir examples/LIBERO/tools/original_images \
    --num_frames 5

EXIT_CODE=$?
echo "=========================================="
echo "脚本执行完成"
echo "退出码: $EXIT_CODE"
echo "时间: $(date)"
echo "=========================================="

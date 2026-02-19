#!/bin/bash
#SBATCH -J libero_sim
#SBATCH -p h200n
#SBATCH -A prj0000000267
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1          # 必须要有GPU
#SBATCH -o sim_object.out            # 日志直接输出到当前目录
#SBATCH -e sim_object.err

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

# 单个文件测试（使用-u参数使Python输出unbuffered，2>&1合并stderr到stdout）
# task_id和episode_idx现在会自动从文件推断，不需要手动指定
python -u examples/LIBERO/tools/precompute_libero_bbox_from_lerobot.py \
  --parquet_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000.parquet \
  --task_suite_name libero_10 \
  --output_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000_with_seg.parquet \
  --resolution 256 \
  --seed 0 2>&1 

EXIT_CODE=$?
echo "=========================================="
echo "脚本执行完成"
echo "退出码: $EXIT_CODE"
echo "时间: $(date)"
echo "=========================================="

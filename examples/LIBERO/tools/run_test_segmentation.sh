#!/bin/bash
#SBATCH -J libero_sim
#SBATCH -p h200n
#SBATCH -A prj0000000267
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1          # 必须要有GPU
#SBATCH -o slurm.out        
#SBATCH -e slurm.err     

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


python examples/LIBERO/tools/test_parquet_segmentation.py \
    --parquet_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000_with_seg.parquet \
    --num_steps 200 \
    --output_prefix episode_10_000000

echo ""
echo "✅ 测试完成！"
echo "📂 查看结果: examples/LIBERO/tools/test_segmentation_output/"

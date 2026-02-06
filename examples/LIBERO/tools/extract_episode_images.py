#!/usr/bin/env python3
"""
从LIBERO环境渲染episode的前N帧图像（两个视角）
用法: python extract_episode_images.py episode_000096.parquet --num_frames 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

# Add LIBERO to Python path
LIBERO_HOME = os.environ.get("LIBERO_HOME", "/home/users/astar/i2r/lishijie/yk/LIBERO")
if LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)

# Patch torch loading
import torch
torch.load = partial(torch.load, weights_only=False)

# Force EGL rendering
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def auto_get_task_params(parquet_path: Path, task_suite: benchmark.Benchmark):
    """从parquet文件名自动推断task_id和episode_idx

    注意：任务的自然语言描述顺序在LeRobot导出的数据里与LIBERO内置task顺序不同，
    不能直接使用tasks.jsonl里的task_index。这里用语言描述匹配Benchmark里的task.language，
    保证选择正确的bddl和初始状态。"""
    filename = parquet_path.stem
    global_idx = int(filename.split('_')[-1])

    base_dir = parquet_path.parent.parent.parent
    episodes_jsonl = base_dir / "meta" / "episodes.jsonl"
    tasks_jsonl = base_dir / "meta" / "tasks.jsonl"

    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"找不到episodes.jsonl: {episodes_jsonl}")
    if not tasks_jsonl.exists():
        raise FileNotFoundError(f"找不到tasks.jsonl: {tasks_jsonl}")

    # 用benchmark中的语言描述构建映射，避免依赖数据集自带的task_index顺序
    suite_lang_to_id = {task.language: idx for idx, task in enumerate(task_suite.tasks)}

    # 读取episodes.jsonl
    with open(episodes_jsonl) as f:
        episodes = [json.loads(line) for line in f]

    if global_idx >= len(episodes):
        raise ValueError(f"Episode {global_idx} 不存在（总共{len(episodes)}个episodes）")

    task_desc = episodes[global_idx]['tasks'][0]

    if task_desc not in suite_lang_to_id:
        raise ValueError(
            "任务描述在benchmark中未匹配到，可能是任务套件名称不一致或描述有差异: "
            f"{task_desc}"
        )
    task_id = suite_lang_to_id[task_desc]

    # 对所有episodes按任务分组
    task_episodes = defaultdict(list)
    for idx, ep in enumerate(episodes):
        task_episodes[ep['tasks'][0]].append(idx)

    episode_list = task_episodes[task_desc]
    episode_idx = episode_list.index(global_idx)

    return task_id, episode_idx, task_desc


def get_libero_env(task_bddl_file: Path, resolution: int = 256, seed: int = 0):
    """初始化LIBERO环境"""
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def extract_images(
    parquet_path: str,
    num_frames: int = 3,
    output_dir: str = None,
    task_suite_name: str = "libero_spatial",
    resolution: int = 256,
    seed: int = 0,
):
    """
    从LIBERO环境渲染episode的前N帧图像
    
    Args:
        parquet_path: parquet文件路径
        num_frames: 提取的帧数
        output_dir: 输出目录
        task_suite_name: 任务套件名称
        resolution: 图像分辨率
        seed: 随机种子
    """
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        print(f"❌ 错误: 文件不存在: {parquet_path}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path("episode_images") / parquet_path.stem
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 读取parquet文件: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    total_frames = len(df)
    print(f"   总帧数: {total_frames}")
    
    # 确定实际提取的帧数
    num_frames = min(num_frames, total_frames)
    print(f"   提取前 {num_frames} 帧")
    
    # 初始化LIBERO benchmark（需要先拿到task_suite来做语言匹配）
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    # 自动推断任务参数
    print(f"\n🔍 推断任务参数...")
    task_id, episode_idx, task_desc = auto_get_task_params(parquet_path, task_suite)
    print(f"   任务描述: {task_desc}")
    print(f"   task_id: {task_id}, episode_idx: {episode_idx}")
    
    # 初始化LIBERO环境
    print(f"\n🚀 初始化LIBERO环境...")
    task = task_suite.get_task(task_id)
    
    task_bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    
    env = get_libero_env(task_bddl_file, resolution, seed)
    
    # 设置初始状态
    env.reset()
    initial_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(initial_states[episode_idx])
    
    print(f"\n📸 开始渲染图像...")
    saved_files = []
    
    # 保存第0帧（初始状态）
    agentview_img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    
    # 保存agentview
    img = Image.fromarray(agentview_img)
    save_path = output_dir / f"frame_000_agentview.png"
    img.save(save_path)
    saved_files.append(save_path)
    print(f"   ✓ Frame 0 - Agentview: {save_path.name} ({agentview_img.shape})")
    
    # 保存wrist
    img = Image.fromarray(wrist_img)
    save_path = output_dir / f"frame_000_wrist.png"
    img.save(save_path)
    saved_files.append(save_path)
    print(f"   ✓ Frame 0 - Wrist: {save_path.name} ({wrist_img.shape})")
    
    # 执行动作并保存后续帧
    for frame_idx in range(1, num_frames):
        # 从parquet读取动作
        action_data = df.iloc[frame_idx]
        
        action = np.array([
            action_data.get("action.delta_eef_position.x", 0.0),
            action_data.get("action.delta_eef_position.y", 0.0),
            action_data.get("action.delta_eef_position.z", 0.0),
            action_data.get("action.delta_eef_axis_angle.x", 0.0),
            action_data.get("action.delta_eef_axis_angle.y", 0.0),
            action_data.get("action.delta_eef_axis_angle.z", 0.0),
            action_data.get("action.gripper_position", -1.0),
        ], dtype=np.float32)
        
        # 执行动作
        obs, reward, done, info = env.step(action.tolist())
        
        # 保存图像
        agentview_img = obs["agentview_image"]
        wrist_img = obs["robot0_eye_in_hand_image"]
        
        # Agentview
        img = Image.fromarray(agentview_img)
        save_path = output_dir / f"frame_{frame_idx:03d}_agentview.png"
        img.save(save_path)
        saved_files.append(save_path)
        print(f"   ✓ Frame {frame_idx} - Agentview: {save_path.name}")
        
        # Wrist
        img = Image.fromarray(wrist_img)
        save_path = output_dir / f"frame_{frame_idx:03d}_wrist.png"
        img.save(save_path)
        saved_files.append(save_path)
        print(f"   ✓ Frame {frame_idx} - Wrist: {save_path.name}")
    
    env.close()
    
    print(f"\n✅ 完成！共保存 {len(saved_files)} 张图像")
    print(f"📁 输出目录: {output_dir.absolute()}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="从LIBERO环境渲染episode的前N帧图像"
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="parquet文件路径",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="提取的帧数（默认3帧）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认为当前目录下的episode_images/）",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO任务套件名称",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="图像分辨率",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    
    args = parser.parse_args()
    
    extract_images(
        parquet_path=args.parquet_path,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        task_suite_name=args.task_suite_name,
        resolution=args.resolution,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
批量处理LIBERO数据集的segmentation信息
支持多个任务套件的并行处理，并生成ID-Label映射文件
"""

import os
import sys
import json
import argparse
import pathlib
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

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

from libero.libero import benchmark

# Import processing function
from precompute_libero_bbox_from_lerobot import process_trajectory, auto_get_task_params


def collect_id_to_label_mappings(
    base_dir: pathlib.Path,
    task_suite_name: str,
    output_json_path: pathlib.Path
) -> Dict:
    """
    收集指定任务套件所有task的ID到Label映射
    
    Args:
        base_dir: 数据集根目录
        task_suite_name: 任务套件名称
        output_json_path: 输出JSON文件路径
        
    Returns:
        映射字典: {task_suite_name: {task_id: {id: label, ...}, ...}}
    """
    print(f"\n{'='*60}")
    print(f"收集 {task_suite_name} 的ID映射...")
    print(f"{'='*60}")
    
    # 导入必要的LIBERO模块
    import re
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    suite_mappings = {}
    
    # 收集所有task的映射
    for task_id in range(len(task_suite.tasks)):
        task = task_suite.get_task(task_id)
        print(f"\n  Task {task_id}: {task.language}")
        
        # 初始化环境
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_names": ["agentview"],
            "camera_segmentations": "instance",
        }
        
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        env.reset()
        
        # 构建ID到Label的映射
        id_to_label = {}
        for i, instance_name in enumerate(list(env.env.model.instances_to_ids.keys())):
            # 过滤机械臂相关物体（包括Panda、Mount、Gripper等）
            if any(keyword in instance_name for keyword in ["Panda", "Mount", "Gripper"]):
                continue
            # 格式化名称
            formatted_name = re.sub(r'_\d+$', '', instance_name)
            formatted_name = formatted_name.replace('_', ' ')
            id_to_label[str(i + 1)] = formatted_name
        
        suite_mappings[str(task_id)] = id_to_label
        print(f"    映射: {id_to_label}")
        
        env.close()
    
    return {task_suite_name: suite_mappings}


def process_single_file(
    parquet_path: pathlib.Path,
    task_suite_name: str,
    output_dir: pathlib.Path,
    resolution: int = 256,
    seed: int = 0
) -> Tuple[bool, str]:
    """
    处理单个parquet文件
    
    Returns:
        (success, message)
    """
    try:
        # 自动推断task_id和episode_idx
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        task_id, episode_idx = auto_get_task_params(parquet_path, task_suite)
        
        # 输出路径
        output_path = output_dir / parquet_path.name
        
        # 处理文件
        process_trajectory(
            parquet_path=parquet_path,
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            output_path=output_path,
            resolution=resolution,
            seed=seed,
        )
        
        return True, f"✓ {parquet_path.name}"
    except Exception as e:
        return False, f"✗ {parquet_path.name}: {str(e)}"


def process_task_suite(
    base_dir: pathlib.Path,
    task_suite_name: str,
    max_workers: int = 4,
    resolution: int = 256,
    seed: int = 0
) -> Dict[str, int]:
    """
    处理单个任务套件的所有文件
    
    Returns:
        统计信息: {"total": N, "success": M, "failed": K}
    """
    print(f"\n{'='*60}")
    print(f"处理任务套件: {task_suite_name}")
    print(f"{'='*60}")
    
    # 查找数据目录
    suite_dir = base_dir / f"{task_suite_name}_no_noops_1.0.0_lerobot"
    chunk_000_dir = suite_dir / "data" / "chunk-000"
    
    if not chunk_000_dir.exists():
        print(f"  警告: 未找到数据目录 {chunk_000_dir}")
        return {"total": 0, "success": 0, "failed": 0}
    
    # 创建输出目录
    chunk_001_dir = suite_dir / "data" / "chunk-001"
    chunk_001_dir.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {chunk_001_dir}")
    
    # 查找所有parquet文件
    parquet_files = sorted(chunk_000_dir.glob("episode_*.parquet"))
    print(f"  找到 {len(parquet_files)} 个文件")
    
    if len(parquet_files) == 0:
        return {"total": 0, "success": 0, "failed": 0}
    
    # 处理文件
    stats = {"total": len(parquet_files), "success": 0, "failed": 0}
    
    if max_workers > 1:
        # 多进程处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    parquet_path,
                    task_suite_name,
                    chunk_001_dir,
                    resolution,
                    seed
                ): parquet_path
                for parquet_path in parquet_files
            }
            
            with tqdm(total=len(parquet_files), desc=f"  处理 {task_suite_name}") as pbar:
                for future in as_completed(futures):
                    success, message = future.result()
                    if success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                        print(f"\n  {message}")
                    pbar.update(1)
    else:
        # 单进程处理（便于调试）
        for parquet_path in tqdm(parquet_files, desc=f"  处理 {task_suite_name}"):
            success, message = process_single_file(
                parquet_path,
                task_suite_name,
                chunk_001_dir,
                resolution,
                seed
            )
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                print(f"\n  {message}")
    
    print(f"\n  统计: 成功 {stats['success']}/{stats['total']}, 失败 {stats['failed']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="批量处理LIBERO数据集的segmentation信息"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/users/astar/i2r/lishijie/yk/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA",
        help="数据集根目录",
    )
    parser.add_argument(
        "--task_suites",
        type=str,
        nargs="+",
        default=["libero_spatial", "libero_goal", "libero_object", "libero_10"],
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="要处理的任务套件列表",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="并行处理的进程数（建议1-4，避免GPU内存不足）",
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
    parser.add_argument(
        "--mappings_output",
        type=str,
        default="/home/users/astar/i2r/lishijie/yk/starVLA/examples/LIBERO/tools/libero_id_to_label_mappings.json",
        help="ID-Label映射JSON文件输出路径",
    )
    parser.add_argument(
        "--collect_mappings_only",
        action="store_true",
        help="仅收集ID映射，不处理数据",
    )
    
    args = parser.parse_args()
    
    base_dir = pathlib.Path(args.base_dir)
    mappings_output = pathlib.Path(args.mappings_output)
    
    print("="*60)
    print("LIBERO 批量处理工具")
    print("="*60)
    print(f"数据根目录: {base_dir}")
    print(f"任务套件: {', '.join(args.task_suites)}")
    print(f"并行进程数: {args.max_workers}")
    print(f"映射输出: {mappings_output}")
    print("="*60)
    
    # 收集ID映射
    all_mappings = {}
    for task_suite_name in args.task_suites:
        suite_mappings = collect_id_to_label_mappings(
            base_dir,
            task_suite_name,
            mappings_output
        )
        all_mappings.update(suite_mappings)
    
    # 保存映射JSON
    with open(mappings_output, "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, indent=2, ensure_ascii=False)
    print(f"\n✅ ID映射已保存到: {mappings_output}")
    
    # 如果只收集映射，则退出
    if args.collect_mappings_only:
        print("\n仅收集映射模式，跳过数据处理")
        return
    
    # 处理数据文件
    total_stats = {"total": 0, "success": 0, "failed": 0}
    
    for task_suite_name in args.task_suites:
        stats = process_task_suite(
            base_dir,
            task_suite_name,
            max_workers=args.max_workers,
            resolution=args.resolution,
            seed=args.seed
        )
        total_stats["total"] += stats["total"]
        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
    
    # 最终统计
    print("\n" + "="*60)
    print("批量处理完成")
    print("="*60)
    print(f"总文件数: {total_stats['total']}")
    print(f"成功: {total_stats['success']}")
    print(f"失败: {total_stats['failed']}")
    print(f"成功率: {total_stats['success']/total_stats['total']*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()

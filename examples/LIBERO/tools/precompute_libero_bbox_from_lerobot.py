#!/usr/bin/env python3
"""
从LeRobot格式的parquet文件中预计算LIBERO的bbox和mask信息
读取轨迹数据，在LIBERO环境中重放，提取segmentation，并将结果存回parquet
适配 PaDT (Patch-aware Deformable Transformer) 的数据逻辑:
1. Patch计算模拟 644px 长边缩放
2. BBox 使用归一化坐标
"""

import os
import sys
import json
import argparse
import pathlib
from functools import partial
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from pycocotools import mask as maskUtils
from tqdm import tqdm

# Add LIBERO to Python path
LIBERO_HOME = os.environ.get("LIBERO_HOME", "/home/users/astar/i2r/lishijie/yk/LIBERO")
print(f"LIBERO_HOME: {LIBERO_HOME}", flush=True)
if LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)
    print(f"已添加LIBERO到sys.path", flush=True)

print("导入依赖库完成", flush=True)

# Patch torch loading
import torch
torch.load = partial(torch.load, weights_only=False)

# Force EGL rendering
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

print("开始导入LIBERO模块...", flush=True)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
print("LIBERO模块导入完成", flush=True)


def auto_get_task_params(parquet_path: pathlib.Path, task_suite: benchmark.Benchmark) -> Tuple[int, int]:
    """
    从parquet文件名自动推断task_id和episode_idx

    注意：LeRobot导出的tasks.jsonl里的task_index顺序与LIBERO内部的task顺序不一致，
    不能直接用task_index。这里用自然语言描述匹配Benchmark中的task.language，
    保证选到正确的bddl和初始状态。
    """
    filename = parquet_path.stem  # "episode_000096"
    global_idx = int(filename.split('_')[-1])

    base_dir = parquet_path.parent.parent.parent
    episodes_jsonl = base_dir / "meta" / "episodes.jsonl"
    tasks_jsonl = base_dir / "meta" / "tasks.jsonl"

    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"找不到episodes.jsonl: {episodes_jsonl}")
    if not tasks_jsonl.exists():
        raise FileNotFoundError(f"找不到tasks.jsonl: {tasks_jsonl}")

    # 用benchmark的语言描述映射，避免依赖数据集自带的task_index
    suite_lang_to_id = {task.language: idx for idx, task in enumerate(task_suite.tasks)}

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

    task_episodes = defaultdict(list)
    for idx, ep in enumerate(episodes):
        task_episodes[ep['tasks'][0]].append(idx)

    episode_list = task_episodes[task_desc]
    episode_idx = episode_list.index(global_idx)

    print(f"  自动推断参数:")
    print(f"    - Episode全局索引: {global_idx}")
    print(f"    - 任务描述: {task_desc}")
    print(f"    - task_id: {task_id}")
    print(f"    - episode_idx: {episode_idx} (该任务共{len(episode_list)}个episodes)")

    return task_id, episode_idx


def extract_bbox_and_mask(seg_image: np.ndarray, id_to_label: Dict[int, str], min_pixels: int = 50) -> Dict[str, Dict]:
    """
    从segmentation图像中提取信息，逻辑适配 PaDT 模型。
    
    Args:
        seg_image: segmentation图像 (H, W) 或 (H, W, C)
        id_to_label: ID到物体名称的映射字典
        min_pixels: 最小像素数阈值
        
    Returns:
        字典: {uid: {"label": str, "bbox": [x1, y1, x2, y2], "patches": [...], "mask": {...}}}
    """
    # 确保是2D数组
    if seg_image.ndim == 3:
        seg_image = seg_image[:, :, 0]
    
    H, W = seg_image.shape  # 原始尺寸 (通常是 256x256)
    seg_image = seg_image.astype(np.int32)
    unique_ids = np.unique(seg_image)
    
    results = {}
    
    # -----------------------------------------------------------
    # PaDT 核心逻辑配置
    # -----------------------------------------------------------
    # PaDT 推理/训练时会将图片长边 resize 到 644
    PADT_MAX_SIDE = 644
    PADT_PATCH_SIZE = 28
    
    # 1. 计算 PaDT 缩放后的尺寸
    scale = PADT_MAX_SIDE / max(H, W)
    target_w = int(W * scale)
    target_h = int(H * scale)
    
    # 2. 计算 PaDT Grid 尺寸 (例如 644/28 = 23)
    grid_w = round(target_w / PADT_PATCH_SIZE)
    grid_h = round(target_h / PADT_PATCH_SIZE)
    
    for uid in unique_ids:
        uid = int(uid)
        if uid == 0:  # 跳过背景
            continue
        
        # 只处理在映射中的物体（自动过滤机械臂等）
        if uid not in id_to_label:
            continue
        
        # 生成原始分辨率的二值 mask
        binary_mask = (seg_image == uid).astype(np.uint8)
        pixel_count = np.sum(binary_mask)
        
        # 过滤小噪点
        if pixel_count < min_pixels:
            continue
        
        # -------------------------------------------------------
        # A. 计算归一化 BBox [x1/W, y1/H, x2/W, y2/H]
        # (基于原始尺寸计算，归一化后是分辨率无关的)
        # -------------------------------------------------------
        rows, cols = np.where(binary_mask > 0)
        y_min, y_max = float(np.min(rows)), float(np.max(rows))
        x_min, x_max = float(np.min(cols)), float(np.max(cols))
        
        bbox_normalized = [
            x_min / W,
            y_min / H,
            x_max / W, 
            y_max / H
        ]
        
        # -------------------------------------------------------
        # B. 计算 Patches (PaDT 逻辑)
        # 必须模拟 PaDT 的 resize 过程来计算 patches 索引
        # -------------------------------------------------------
        # 1. 将 mask 缩放到 PaDT 的输入尺寸 (e.g. 644x644)
        # 使用 INTER_NEAREST 保持二值特性，或者 INTER_LINEAR 后阈值化
        resized_mask_large = cv2.resize(
            binary_mask.astype(np.uint8), 
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 2. 将 mask 缩放到 Grid 尺寸 (e.g. 23x23)
        # 这里使用 INTER_AREA 进行下采样，得到每个 Patch 的覆盖率 (0~1)
        patch_coverage = cv2.resize(
            resized_mask_large, 
            (grid_w, grid_h), 
            interpolation=cv2.INTER_AREA
        )
        
        # 3. 提取 Active Patches
        # 只要 Patch 里有物体 (覆盖率 > 0)，就认为是 Active
        # 为了容错，可以设置一个极小的阈值比如 0.05 (5% 覆盖)
        active_mask = patch_coverage > 0.01 
        
        # 4. 获取扁平化的索引列表
        # reshape(-1) 拉平，np.where 找 True 的位置
        active_indices = np.where(active_mask.reshape(-1))[0].tolist()
        
        # -------------------------------------------------------
        # C. 计算 RLE mask (COCO格式)
        # (保持原始分辨率存储，训练器会负责 resize)
        # -------------------------------------------------------
        fortran_binary_mask = np.asfortranarray(binary_mask)
        rle = maskUtils.encode(fortran_binary_mask)
        counts_str = rle['counts'].decode('utf-8')
        
        # 获取对应的 label
        label = id_to_label.get(uid, f"unknown_object_{uid}")
        
        results[str(uid)] = {
            "bbox": bbox_normalized,
            "patches": active_indices, # 这里的 indices 对应的是 23x23 这种大 Grid
            "mask": {
                "size": rle['size'],  
                "counts": counts_str,
                "label": label,
            }
        }
    
    return results


def get_libero_env(task_bddl_file: pathlib.Path, resolution: int = 256, seed: int = 0):
    """初始化LIBERO环境"""
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_segmentations": "instance",
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def process_trajectory(
    parquet_path: pathlib.Path,
    task_suite_name: str,
    task_id: int,
    episode_idx: int = None,  # 现在可选，优先从parquet获取初始状态
    output_path: pathlib.Path = None,
    resolution: int = 256,
    seed: int = 0,
) -> pd.DataFrame:
    """
    处理单个轨迹文件，提取segmentation信息并添加到parquet
    """
    print(f"Processing: {parquet_path}")
    
    # 读取parquet数据
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} steps")
    
    # 初始化LIBERO环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    
    env = get_libero_env(task_bddl_file, resolution, seed)
    
    # 重置环境并设置初始状态
    env.reset()
    
    # 构建 ID 到实例名的映射（自动从环境获取）
    import re
    id_to_label = {}
    for i, instance_name in enumerate(list(env.env.model.instances_to_ids.keys())):
        # 过滤机械臂相关物体（通过关键词匹配，包括Panda、Mount、Gripper等）
        if any(keyword in instance_name for keyword in ["Panda", "Mount", "Gripper"]):
            continue
        # segmentation 图像中的像素值 = 枚举索引 + 1
        # 格式化名称：去掉数字后缀，下划线替换为空格
        formatted_name = re.sub(r'_\d+$', '', instance_name)  # 去掉 _1, _2 等数字后缀
        formatted_name = formatted_name.replace('_', ' ')      # 下划线替换为空格
        id_to_label[i + 1] = formatted_name
    
    print(f"  构建ID映射: {id_to_label}")
    
    # 使用预定义的初始状态
    # 注意：parquet中的observation.state只有7维（机器人状态），不包含物体位置
    # 必须使用LIBERO预定义的完整初始状态
    if episode_idx is None:
        raise ValueError(
            "必须提供 --episode_idx 参数！\n"
            "注意：episode_idx 是该task内部的相对索引（0开始），不是episode文件名中的数字。\n"
            "例如：episode_000096.parquet 可能需要 --episode_idx 10（取决于该episode在其task中的顺序）"
        )
    
    print(f"  使用预定义初始状态 (episode_idx={episode_idx})")
    initial_states = task_suite.get_task_init_states(task_id)
    if episode_idx >= len(initial_states):
        raise ValueError(
            f"episode_idx {episode_idx} 超出范围，task_id={task_id} 只有 {len(initial_states)} 个初始状态。"
            f"\n提示：episode_idx应该是该task内部的索引（0到{len(initial_states)-1}），不是全局episode编号。"
        )
    obs = env.set_init_state(initial_states[episode_idx])
    
    # 存储每一步的segmentation信息
    agentview_seg_info_list = []
    wrist_seg_info_list = []
    
    # 第一步：使用初始状态
    agentview_seg = obs.get("agentview_segmentation_instance")
    wrist_seg = obs.get("robot0_eye_in_hand_segmentation_instance")
    
    if agentview_seg is not None:
        # IMPORTANT: 旋转180度以匹配训练预处理
        agentview_seg_rotated = np.ascontiguousarray(agentview_seg[::-1, ::-1])
        agentview_info = extract_bbox_and_mask(agentview_seg_rotated, id_to_label) 
    else:
        agentview_info = {}
    
    if wrist_seg is not None:
        wrist_seg_rotated = np.ascontiguousarray(wrist_seg[::-1, ::-1])
        wrist_info = extract_bbox_and_mask(wrist_seg_rotated, id_to_label)
    else:
        wrist_info = {}
    
    agentview_seg_info_list.append(json.dumps(agentview_info))
    wrist_seg_info_list.append(json.dumps(wrist_info))
    
    # 后续步骤：执行动作并获取segmentation
    for idx in tqdm(range(1, len(df)), desc="  Replaying trajectory"):
        # 从parquet读取动作
        action_data = df.iloc[idx]
        
        # 构建7维动作 [x, y, z, rx, ry, rz, gripper]
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
        
        # 提取segmentation信息
        agentview_seg = obs.get("agentview_segmentation_instance")
        wrist_seg = obs.get("robot0_eye_in_hand_segmentation_instance")
        
        if agentview_seg is not None:
            agentview_seg_rotated = np.ascontiguousarray(agentview_seg[::-1, ::-1])
            agentview_info = extract_bbox_and_mask(agentview_seg_rotated, id_to_label)
        else:
            agentview_info = {}
        
        if wrist_seg is not None:
            wrist_seg_rotated = np.ascontiguousarray(wrist_seg[::-1, ::-1])
            wrist_info = extract_bbox_and_mask(wrist_seg_rotated, id_to_label)
        else:
            wrist_info = {}
        
        agentview_seg_info_list.append(json.dumps(agentview_info))
        wrist_seg_info_list.append(json.dumps(wrist_info))
    
    env.close()
    
    # 添加新列到DataFrame
    df["segmentation.agentview_bbox_mask"] = agentview_seg_info_list
    df["segmentation.wrist_bbox_mask"] = wrist_seg_info_list
    
    # 保存结果
    if output_path is None:
        output_path = parquet_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="从LeRobot parquet文件预计算LIBERO segmentation信息 (PaDT Compatible)"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="/home/users/astar/i2r/lishijie/yk/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/data/chunk-000/episode_000096.parquet",
        help="输入parquet文件路径",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO任务套件名称",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="任务ID（如果不提供则自动从文件推断）",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=None,
        help="Episode在该task中的相对索引（如果不提供则自动从文件推断）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出parquet文件路径（默认覆盖原文件）",
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
    
    parquet_path = pathlib.Path(args.parquet_path)
    output_path = pathlib.Path(args.output_path) if args.output_path else None
    
    if not parquet_path.exists():
        print(f"错误: 文件不存在: {parquet_path}")
        sys.exit(1)
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    # 自动推断task_id和episode_idx（如果未提供）
    if args.task_id is None or args.episode_idx is None:
        print("正在自动推断task_id和episode_idx...")
        task_id, episode_idx = auto_get_task_params(parquet_path, task_suite)
    else:
        task_id = args.task_id
        episode_idx = args.episode_idx
    print("task_id:", task_id)
    print("episode_idx:", episode_idx)
    # 处理轨迹
    df = process_trajectory(
        parquet_path=parquet_path,
        task_suite_name=args.task_suite_name,
        task_id=task_id,
        episode_idx=episode_idx,
        output_path=output_path,
        resolution=args.resolution,
        seed=args.seed,
    )
    
    print("\n✅ 处理完成！")
    print(f"新增列:")
    print(f"  - segmentation.agentview_bbox_mask")
    print(f"  - segmentation.wrist_bbox_mask")
    print(f"\n数据格式示例 (ID key下包含 patches, bbox(0-1), mask):")
    if len(df) > 0:
        sample_data = json.loads(df["segmentation.agentview_bbox_mask"].iloc[0])
        if sample_data:
            sample_id = list(sample_data.keys())[0]
            print(f"  ID {sample_id}: {sample_data[sample_id].keys()}")
            print(f"  BBox sample: {sample_data[sample_id]['bbox']}")
            print(f"  Patches count: {len(sample_data[sample_id]['patches'])}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
根据episode文件名自动计算正确的task_id和episode_idx参数
用法: python get_episode_params.py episode_000096.parquet
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def get_episode_params(episode_file: str, episodes_jsonl_path: str, tasks_jsonl_path: str = None):
    """
    从episode文件名解析出正确的参数
    
    Args:
        episode_file: 例如 "episode_000096.parquet"
        episodes_jsonl_path: episodes.jsonl文件路径
        tasks_jsonl_path: tasks.jsonl文件路径（可选，如果None则自动推断）
    
    Returns:
        (global_idx, task_id, episode_idx_in_task, task_desc, episode_list)
    """
    # 从文件名提取全局索引
    filename = Path(episode_file).stem  # "episode_000096"
    global_idx = int(filename.split('_')[-1])  # 96
    
    # 自动推断tasks.jsonl路径
    if tasks_jsonl_path is None:
        episodes_path = Path(episodes_jsonl_path)
        tasks_jsonl_path = episodes_path.parent / "tasks.jsonl"
    
    # 读取tasks.jsonl，建立任务描述到task_id的映射
    task_desc_to_id = {}
    with open(tasks_jsonl_path) as f:
        for line in f:
            task_data = json.loads(line)
            task_desc_to_id[task_data['task']] = task_data['task_index']
    
    # 读取episodes.jsonl
    with open(episodes_jsonl_path) as f:
        episodes = [json.loads(line) for line in f]
    
    if global_idx >= len(episodes):
        raise ValueError(f"Episode {global_idx} 不存在（总共{len(episodes)}个episodes）")
    
    # 获取该episode的任务描述
    task_desc = episodes[global_idx]['tasks'][0]
    
    # 从tasks.jsonl中查找task_id
    if task_desc not in task_desc_to_id:
        raise ValueError(f"任务描述未在tasks.jsonl中找到: {task_desc}")
    task_id = task_desc_to_id[task_desc]
    
    # 对所有episodes按任务分组
    task_episodes = defaultdict(list)
    for idx, ep in enumerate(episodes):
        task_episodes[ep['tasks'][0]].append(idx)
    
    # 找到该episode在该任务中的相对索引
    episode_list = task_episodes[task_desc]
    episode_idx_in_task = episode_list.index(global_idx)
    
    return global_idx, task_id, episode_idx_in_task, task_desc, episode_list


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python get_episode_params.py <episode_file> [--suite libero_spatial]")
        print("示例: python get_episode_params.py episode_000096.parquet")
        sys.exit(1)
    
    episode_file = sys.argv[1]
    
    # 确定数据集路径
    suite_name = "libero_spatial"  # 默认
    if "--suite" in sys.argv:
        suite_idx = sys.argv.index("--suite")
        suite_name = sys.argv[suite_idx + 1]
    
    base_dir = Path(__file__).parent.parent.parent.parent
    episodes_jsonl_path = base_dir / "playground/Datasets/LEROBOT_LIBERO_DATA" / f"{suite_name}_no_noops_1.0.0_lerobot/meta/episodes.jsonl"
    tasks_jsonl_path = base_dir / "playground/Datasets/LEROBOT_LIBERO_DATA" / f"{suite_name}_no_noops_1.0.0_lerobot/meta/tasks.jsonl"
    
    if not episodes_jsonl_path.exists():
        print(f"❌ 错误: 找不到 {episodes_jsonl_path}")
        sys.exit(1)
    
    if not tasks_jsonl_path.exists():
        print(f"❌ 错误: 找不到 {tasks_jsonl_path}")
        sys.exit(1)
    
    try:
        global_idx, task_id, episode_idx, task_desc, episode_list = get_episode_params(
            episode_file, episodes_jsonl_path, tasks_jsonl_path
        )
        
        print(f"📄 Episode文件: {episode_file}")
        print(f"🌐 全局索引: {global_idx}")
        print(f"📋 任务描述: {task_desc}")
        print(f"🎯 task_id: {task_id}")
        print(f"📍 episode_idx: {episode_idx}")
        print(f"📊 该任务共有 {len(episode_list)} 个episodes")
        print(f"📝 该任务的所有episode编号: {episode_list[:10]}{'...' if len(episode_list) > 10 else ''}")
        print()
        print("✅ 使用以下参数运行脚本:")
        print(f"   --task_id {task_id} --episode_idx {episode_idx}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

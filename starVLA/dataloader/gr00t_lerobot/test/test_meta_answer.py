import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[4]  # 向上4级到项目根目录
sys.path.insert(0, str(project_root))

# 避免导入 VLM 依赖（transformers/huggingface），仅测试 answers 加载
os.environ["STARVLA_SKIP_VLM_IMPORTS"] = "1"

# 直接导入底层类，避免触发模型相关的依赖
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG

# 数据集路径（相对于项目根目录）
data_root = project_root / 'playground' / 'Datasets' / 'LEROBOT_LIBERO_DATA'

print(f"项目根目录: {project_root}")
print(f"数据根目录: {data_root}")
print("="*60)

try:
    # 手动创建数据集（不使用 make_LeRobotSingleDataset，避免依赖冲突）
    robot_type = 'libero_franka'
    dataset_name = 'libero_10_no_noops_1.0.0_lerobot'
    dataset_path = data_root / dataset_name
    
    # 获取配置
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    # 创建数据集
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
        delete_pause_frame=False,
        data_cfg={"lerobot_version": "v2.0"}
    )
    
    print(f"\n✅ 数据集加载成功!")
    print(f"数据集名称: {dataset.dataset_name}")
    print(f"Answers 加载数量: {len(dataset.answers)}")
    
    if not dataset.answers.empty:
        print("\n" + "="*60)
        print("Answers 数据预览:")
        print("="*60)
        print(dataset.answers.head())
        
        print("\n" + "="*60)
        print("第一条 Answer 详情:")
        print("="*60)
        first_answer = dataset.answers.iloc[0]
        print(f"Tasks: {first_answer.get('tasks', [])}")
        print(f"Conversations: {first_answer.get('conversations', [])}")
        print(f"Answer Template: {first_answer.get('answer_template', '')}")
    else:
        print("\n⚠️  Warning: answers.jsonl 文件为空或未找到")
        
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

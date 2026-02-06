# LIBERO Segmentation Precomputation Tools

这套工具用于从LeRobot格式的LIBERO数据集中预计算segmentation信息（bbox和mask），并将结果存储回parquet文件。

## 📁 文件说明

- `precompute_libero_bbox_from_lerobot.py`: 核心脚本，处理单个episode
- `batch_precompute_libero.py`: 批量处理整个数据集
- `verify_precomputed_seg.py`: 验证和可视化处理结果
- `run_precompute.sh`: 快速测试脚本

## 🚀 使用方法

### 1. 环境要求

- 必须在LIBERO conda环境中运行
- 确保已安装依赖：`pycocotools`, `pandas`, `pillow`
- **重要**: 需要设置LIBERO路径到Python path

```bash
conda activate libero_env  # 激活LIBERO环境
pip install pycocotools pandas pillow tqdm

# 设置环境变量（必需）
export LIBERO_HOME="/home/users/astar/i2r/lishijie/yk/LIBERO"  # 修改为您的LIBERO路径
export PYTHONPATH="${LIBERO_HOME}:${PYTHONPATH:-}"
```

### 2. 处理单个episode

```bash
python examples/LIBERO/tools/precompute_libero_bbox_from_lerobot.py \
    --parquet_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000.parquet \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --episode_idx 0 \
    --resolution 256 \
    --seed 0
```

**参数说明：**
- `--parquet_path`: 输入parquet文件路径
- `--task_suite_name`: 任务套件名称（libero_spatial/object/goal/10/90）
- `--task_id`: 任务ID（0-9）
- `--episode_idx`: Episode索引（用于获取LIBERO初始状态）
- `--output_path`: 输出路径（默认覆盖原文件）
- `--resolution`: 图像分辨率（默认256）
- `--seed`: 随机种子（默认0）

### 3. 批量处理数据集

```bash
python examples/LIBERO/tools/batch_precompute_libero.py \
    --dataset_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot \
    --task_suite_name libero_spatial \
    --num_workers 1
```

**注意：** 由于LIBERO环境限制，建议使用单进程（`--num_workers 1`）

### 4. 验证处理结果

```bash
# 检查数据完整性
python examples/LIBERO/tools/verify_precomputed_seg.py \
    --parquet_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000.parquet \
    --check_only

# 可视化特定步骤
python examples/LIBERO/tools/verify_precomputed_seg.py \
    --parquet_path playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000.parquet \
    --step_idx 10 \
    --output_dir verification_output
```

## 📊 输出数据格式

处理后的parquet文件会新增两列：

```python
# 每一列存储JSON字符串，格式如下：
{
    "1": {  # 物体ID
        "bbox": [x1, y1, x2, y2],  # 边界框坐标
        "mask": {
            "size": [256, 256],  # 图像尺寸 [height, width]
            "counts": "aBC123...",  # COCO RLE编码的counts字符串
            "label": "wooden_table"  # 物体名称（从LIBERO环境中获取）
        }
    },
    "2": {...},
    ...
}
```

**新增列名：**
- `segmentation.agentview_bbox_mask`: agentview相机的segmentation信息
- `segmentation.wrist_bbox_mask`: wrist相机的segmentation信息

**注意：** label字段包含物体的实际名称（如"wooden_table"、"red_block"等），从LIBERO环境中自动获取

## 🔍 数据读取示例

```python
import pandas as pd
import json
from pycocotools import mask as maskUtils
import numpy as np

# 读取parquet
df = pd.read_parquet("episode_000000.parquet")

# 获取第10步的agentview segmentation
seg_info = json.loads(df.iloc[10]["segmentation.agentview_bbox_mask"])

# 遍历所有物体
for obj_id, obj_data in seg_info.items():
    bbox = obj_data["bbox"]  # [x1, y1, x2, y2]
    mask_info = obj_data["mask"]  # 包含size、counts和label的字典
    
    # 解码mask为二值图像
    rle_data = {
        'size': mask_info['size'],  # 从mask中读取尺寸
        'counts': mask_info['counts'].encode('utf-8') if isinstance(mask_info['counts'], str) else mask_info['counts']
    }
    binary_mask = maskUtils.decode(rle_data)  # shape: (H, W)
    
    label = mask_info['label']  # 物体标签
    area = np.sum(binary_mask)
    
    print(f"Object {obj_id} (Label: {label}): bbox={bbox}, area={area} pixels")
```

## ⚠️ 注意事项

1. **坐标系一致性**: 已应用180度旋转以匹配训练数据预处理（`[::-1, ::-1]`）
2. **Episode索引**: `--episode_idx`需要与parquet文件名中的episode编号对应
3. **初始状态**: 使用LIBERO的固定初始状态以确保物体位置一致
4. **处理时间**: 每个episode处理时间约为轨迹长度×0.1秒（取决于硬件）
5. **覆盖原文件**: 默认会覆盖原parquet文件，建议先备份

## 🐛 故障排除

### 问题1: 找不到LIBERO模块
```bash
# 确保在LIBERO环境中运行
conda activate libero_env

# 设置LIBERO路径环境变量（关键！）
export LIBERO_HOME="/home/users/astar/i2r/lishijie/yk/LIBERO"
export PYTHONPATH="${LIBERO_HOME}:${PYTHONPATH:-}"
```

**原因**: Python默认不会搜索LIBERO目录，即使conda环境正确。您需要显式将LIBERO添加到PYTHONPATH中。

脚本内部已自动处理此问题（从环境变量读取`LIBERO_HOME`），但如果仍报错，请确保：
1. `LIBERO_HOME` 指向正确的LIBERO安装目录
2. 该目录下有 `libero/libero/` 子目录结构

### 问题2: 渲染错误
```bash
# 确保EGL渲染已启用
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 问题3: Episode索引不匹配
检查`meta/episodes.jsonl`文件，确保`--episode_idx`与实际episode编号一致

## 📝 TODO

- [ ] 自动从parquet文件名推断episode_idx
- [ ] 支持从meta数据自动读取task_id映射
- [ ] 添加进度恢复功能（断点续传）
- [ ] 优化多进程支持

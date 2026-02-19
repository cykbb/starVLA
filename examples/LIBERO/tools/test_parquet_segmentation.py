#!/usr/bin/env python3
"""
验证parquet文件中的bbox和mask信息是否正确
适配 PaDT 逻辑 (644px缩放 -> 23x23 Grid) 的 Patch 可视化
"""

import json
import os
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as maskUtils
import random
from PIL import Image

# 确保能找到相关模块
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "test_segmentation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 可能的图像列名（优先级从高到低）
AGENTVIEW_IMG_COLUMNS = [
    "observation.agentview_image",
    "observation.agentview_rgb",
    "agentview_image",
    "agentview_rgb",
]
WRIST_IMG_COLUMNS = [
    "observation.robot0_eye_in_hand_image",
    "observation.robot0_eye_in_hand_rgb",
    "wrist_image",
    "wrist_rgb",
]


def _select_image_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """从DataFrame中自动选择两路相机的图像列。"""
    def pick(candidates, fallback_contains):
        for name in candidates:
            if name in df.columns:
                return name
        for col in df.columns:
            if all(key in col for key in fallback_contains):
                return col
        return None

    agent_col = pick(AGENTVIEW_IMG_COLUMNS, ["agentview", "image"])
    wrist_col = pick(WRIST_IMG_COLUMNS, ["wrist", "image"])
    if wrist_col is None:
        wrist_col = pick([], ["eye_in_hand", "image"])
    return agent_col, wrist_col


def _cell_to_image(cell):
    """尽量将单元格内容转换为 (H, W, 3) 的uint8 RGB图像。失败返回None。"""
    if cell is None:
        return None
    try:
        if isinstance(cell, str):
            path = Path(cell)
            if path.exists():
                return np.array(Image.open(path).convert("RGB"))
            # 若是JSON数组
            try:
                arr = np.array(json.loads(cell))
            except Exception:
                return None
        elif isinstance(cell, (bytes, bytearray)):
            try:
                with io.BytesIO(cell) as f:
                    return np.array(Image.open(f).convert("RGB"))
            except Exception:
                return None
        else:
            arr = np.array(cell)

        if not isinstance(arr, np.ndarray):
            return None

        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
            return arr.astype(np.uint8)
    except Exception:
        return None
    return None


def decode_and_visualize_step(
    step_idx: int,
    agentview_seg_json: str,
    wrist_seg_json: str,
    output_prefix: str,
    raw_images: dict | None = None,
):
    """
    解码并可视化单个步骤的segmentation信息 (适配 PaDT Grid)
    """
    print(f"\n{'='*60}")
    print(f"📸 测试步骤 {step_idx}")
    print(f"{'='*60}")
    
    agentview_data = json.loads(agentview_seg_json)
    wrist_data = json.loads(wrist_seg_json)
    raw_images = raw_images or {}
    
    # 【核心配置】必须与生成脚本保持一致
    PADT_MAX_SIDE = 644
    PADT_PATCH_SIZE = 28
    
    camera_data = {"agentview": agentview_data, "wrist": wrist_data}
    
    for cam_name, objects in camera_data.items():
        print(f"\n📷 {cam_name} 相机:")
        
        if not objects:
            print("   ⚠ 没有检测到物体")
            continue
            
        first_obj = list(objects.values())[0]
        h, w = first_obj['mask']['size'] # 通常是 256x256
        
        # -----------------------------------------------------------
        # 1. 反推 Grid 结构 (PaDT 逻辑)
        # -----------------------------------------------------------
        scale = PADT_MAX_SIDE / max(h, w)
        target_w = int(w * scale) # ~644
        target_h = int(h * scale) # ~644
        
        grid_cols = int(round(target_w / PADT_PATCH_SIZE)) # ~23
        grid_rows = int(round(target_h / PADT_PATCH_SIZE)) # ~23
        
        print(f"   图像尺寸: {w}x{h}")
        print(f"   PaDT模拟缩放: {target_w}x{target_h}")
        print(f"   Grid网格: {grid_rows}行 x {grid_cols}列 (共 {grid_rows*grid_cols} patches)")
        
        # 计算在原始 256x256 图片上，每个 Patch 应该画多大
        vis_cell_w = w / grid_cols # ~11.1 px
        vis_cell_h = h / grid_rows # ~11.1 px

        # 创建画布
        patches_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        has_raw = cam_name in raw_images and raw_images[cam_name] is not None
        cols = 4 if has_raw else 3
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 8))
        axes = np.atleast_1d(axes).flatten()
        ax_idx = 0
        if has_raw:
            raw_img = raw_images[cam_name]
            if raw_img.shape[-1] == 4:
                raw_img = raw_img[..., :3]
            axes[ax_idx].imshow(raw_img.astype(np.uint8))
            axes[ax_idx].set_title("Raw", fontweight='bold')
            axes[ax_idx].axis('off')
            ax_idx += 1

        ax1 = axes[ax_idx]; ax_idx += 1
        ax2 = axes[ax_idx]; ax_idx += 1
        ax3 = axes[ax_idx]
        
        for obj_id, info in objects.items():
            bbox_norm = info['bbox']
            mask_info = info['mask']
            patches_indices = info.get('patches', []) 
            
            # --- 坐标转换 (BBox 是归一化的，直接乘原图尺寸) ---
            x1 = int(round(bbox_norm[0] * w)); y1 = int(round(bbox_norm[1] * h))
            x2 = int(round(bbox_norm[2] * w)); y2 = int(round(bbox_norm[3] * h))
            x1=max(0,min(w,x1)); y1=max(0,min(h,y1)); x2=max(0,min(w,x2)); y2=max(0,min(h,y2))
            
            # --- 解码 Mask ---
            rle_data = {'size': mask_info['size'], 'counts': mask_info['counts'].encode('utf-8') if isinstance(mask_info['counts'], str) else mask_info['counts']}
            label = mask_info.get('label', f"ID_{obj_id}")
            binary_mask = maskUtils.decode(rle_data)
            
            print(f"   -> 物体 {label}: Patches数量={len(patches_indices)}")

            # --- 可视化准备 ---
            random.seed(int(obj_id) if str(obj_id).isdigit() else hash(obj_id))
            color = np.random.randint(50, 255, 3)
            
            # 1. 绘制 Mask
            for c in range(3):
                mask_canvas[:, :, c] = np.where(binary_mask == 1, color[c], mask_canvas[:, :, c])
            
            # 2. 绘制 BBox
            if x2 > x1 and y2 > y1:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=tuple(color/255.0), facecolor='none', linestyle='--')
                ax2.add_patch(rect)
                ax2.text(x1, max(0, y1-5), label, color='white', fontsize=9, bbox=dict(facecolor=tuple(color/255.0), alpha=0.7))

            # 3. 绘制 Patches (关键修改)
            for p_idx in patches_indices:
                # 算出 Patch 在 Grid 中的行列
                p_row = p_idx // grid_cols
                p_col = p_idx % grid_cols
                
                # 映射回原始图片像素坐标
                px_start = int(p_col * vis_cell_w)
                py_start = int(p_row * vis_cell_h)
                px_end = int((p_col + 1) * vis_cell_w)
                py_end = int((p_row + 1) * vis_cell_h)
                
                # 边界保护
                px_end = min(px_end, w)
                py_end = min(py_end, h)
                
                # 画实心方块
                patches_canvas[py_start:py_end, px_start:px_end] = color

        # --- 显示设置 ---
        ax1.imshow(mask_canvas); ax1.set_title("Mask", fontweight='bold'); ax1.axis('off')
        
        gray_bg = np.ones((h, w, 3), dtype=np.uint8) * 50
        ax2.imshow(gray_bg); ax2.set_title("BBox", fontweight='bold'); ax2.axis('off')

        ax3.imshow(patches_canvas)
        ax3.set_title(f"Patches (mapped from {grid_rows}x{grid_cols} grid)", fontweight='bold')
        
        # 画网格辅助线 (为了看清楚，每隔一个 Patch 画一条)
        for c in range(1, grid_cols):
            x = int(c * vis_cell_w)
            ax3.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
        for r in range(1, grid_rows):
            y = int(r * vis_cell_h)
            ax3.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
        ax3.axis('off')

        save_path = OUTPUT_DIR / f"{output_prefix}_step{step_idx:04d}_{cam_name}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 图片已保存: {save_path}")

def test_parquet_segmentation(
    parquet_path: str,
    num_steps: int = 3,
    output_prefix: str = "test"
):
    """
    测试parquet文件中的segmentation信息
    """
    print(f"\n{'#'*60}")
    print(f"🔍 开始测试Parquet文件的Segmentation信息")
    print(f"{'#'*60}")
    print(f"文件路径: {parquet_path}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if not Path(parquet_path).exists():
        print(f"❌ 错误: 文件不存在: {parquet_path}")
        return
    
    print(f"\n📂 正在读取parquet文件...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ 读取Parquet文件失败: {e}")
        return
        
    print(f"✓ 成功读取，共 {len(df)} 个步骤")
    
    if "segmentation.agentview_bbox_mask" not in df.columns:
        print(f"❌ 错误: 缺少 'segmentation.agentview_bbox_mask' 列")
        return
    
    agent_img_col, wrist_img_col = _select_image_columns(df)
    print("\n图像列自动检测:")
    if agent_img_col:
        print(f"  ✓ agentview 列: {agent_img_col}")
    else:
        print("  ⚠ 未找到 agentview 图像列，将跳过原图可视化")
    if wrist_img_col:
        print(f"  ✓ wrist 列: {wrist_img_col}")
    else:
        print("  ⚠ 未找到 wrist 图像列，将跳过原图可视化")

    test_steps = min(num_steps, len(df))
    print(f"\n开始测试前 {test_steps} 个步骤...")
    
    for step_idx in range(test_steps):
        row = df.iloc[step_idx]
        agentview_seg = row["segmentation.agentview_bbox_mask"]
        wrist_seg = row["segmentation.wrist_bbox_mask"]

        raw_images = {}
        if agent_img_col:
            raw_images["agentview"] = _cell_to_image(row.get(agent_img_col))
        if wrist_img_col:
            raw_images["wrist"] = _cell_to_image(row.get(wrist_img_col))
        
        decode_and_visualize_step(
            step_idx=step_idx,
            agentview_seg_json=agentview_seg,
            wrist_seg_json=wrist_seg,
            output_prefix=output_prefix,
            raw_images=raw_images,
        )
    
    print(f"\n{'#'*60}")
    print(f"✅ 测试完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--output_prefix", type=str, default="episode_000000")
    
    args = parser.parse_args()
    
    test_parquet_segmentation(
        parquet_path=args.parquet_path,
        num_steps=args.num_steps,
        output_prefix=args.output_prefix
    )
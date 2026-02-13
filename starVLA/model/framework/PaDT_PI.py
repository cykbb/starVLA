# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.padt import PaDTForConditionalGeneration
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################

@FRAMEWORK_REGISTRY.register("PaDTPI")
class PaDT_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - PaDT-enhanced Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.padt_vl_interface = get_vlm_model(config=self.config)
        model_cls = PaDTForConditionalGeneration
        
        # Pre-reserve VRT tokens and resize embeddings once BEFORE deepspeed/accelerate wraps the model
        try:
            merge_size = getattr(self.padt_vl_interface.processor, "spatial_merge_size", 2)
            max_vrt_patches = (
                self.config.framework.qwenvl.get("max_vrt_patches", None)
                if hasattr(self.config, "framework") and hasattr(self.config.framework, "qwenvl")
                else None
            )
            if max_vrt_patches is None:
                # conservative default: 4096 visual patches per sample
                max_vrt_patches = 4096
            # grid_thw so that t*h*w / merge_size^2 == max_vrt_patches
            grid_thw = torch.tensor([[1, 1, max_vrt_patches * (merge_size ** 2)]], dtype=torch.int64)
            # add tokens to tokenizer if needed
            self.padt_vl_interface.processor.set_image_grid_thw(grid_thw)
            tok_len = len(self.padt_vl_interface.processor.tokenizer)
            embed_len = self.padt_vl_interface.model.get_input_embeddings().weight.shape[0]
            if tok_len > embed_len:
                self.padt_vl_interface.model.resize_token_embeddings(tok_len, mean_resizing=False)
            if hasattr(self.padt_vl_interface.processor, "model_embed_token_size"):
                self.padt_vl_interface.processor.model_embed_token_size = len(self.padt_vl_interface.processor.tokenizer)
        except Exception as e:
            logger.warning(f"VRT pre-resize failed, training may resize at runtime: {e}")
        

        # dynamic get llm config (Qwen2.5 VL configs sometimes omit top-level hidden_size)
        vl_config = self.padt_vl_interface.model.config
        llm_hidden_size = getattr(vl_config, "hidden_size", None)
        if llm_hidden_size is None and hasattr(vl_config, "text_config"):
            llm_hidden_size = getattr(vl_config.text_config, "hidden_size", None)
        if llm_hidden_size is None:
            model_core = getattr(self.padt_vl_interface.model, "model", None)
            embed_tokens = getattr(model_core, "embed_tokens", None)
            if embed_tokens is not None and hasattr(embed_tokens, "weight"):
                llm_hidden_size = embed_tokens.weight.shape[1]
        if llm_hidden_size is None:
            raise AttributeError("Cannot resolve llm hidden_size from Qwen2_5_VL config/model")

        num_vl_layers = 36
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        

    def compute_vlm_loss(
        self,
        examples: List[dict],
        batch_images: List[List[Image.Image]],
        instructions: List[str],
    ) -> torch.Tensor:
        """
        Compute VLM loss based on answers and segmentation data.
        
        Args:
            examples: List[dict] containing answers and segmentation data
            batch_images: List of image lists [B, [PIL.Image]]
            instructions: List of instruction strings [B]
            
        Returns:
            vlm_loss: torch.Tensor, language modeling loss
        """
        import json
        import re
        import random

        processor = self.padt_vl_interface.processor
        
        # 检查是否有 answers 数据
        if "answers" not in examples[0]:
            return None
            
        # 构建 prompt，收集 completion_raw 与 objects_info，稍后根据 runtime grid 重映射 patches 再生成 completion
        prompts = []
        completion_raw_list = []
        objects_infos_list = []
        patch_ids_all: list[int] = []  # collect offline patch ids to sanity-check with processor grid
        
        # 与离线预处理保持一致：长边 644（PaDT 默认缩放逻辑）
        vlm_image_size = 644
        images_per_sample = []
        vlm_batch_images = []
        for imgs in batch_images:
            if isinstance(imgs, (list, tuple)):
                resized_imgs = [img.resize((vlm_image_size, vlm_image_size)) if hasattr(img, 'resize') else img for img in imgs]
                vlm_batch_images.append(resized_imgs)
                images_per_sample.append(len(imgs))
            else:
                vlm_batch_images.append([imgs.resize((vlm_image_size, vlm_image_size)) if hasattr(imgs, 'resize') else imgs])
                images_per_sample.append(1)
        
        for i_ex, example in enumerate(examples):
            answers_data = example["answers"]
            
            # 从 answers 中提取对话
            if "conversations" in answers_data and len(answers_data["conversations"]) > 0:
                # 使用对话作为 prompt，同时加入图片占位（与 build_padtvl_inputs 对齐）
                conversation = answers_data["conversations"][0]
                imgs = vlm_batch_images[i_ex]  # 已 resize 的低分辨率图
                if isinstance(imgs, (list, tuple)):
                    content = [{"type": "image", "image": img} for img in imgs]
                else:
                    content = [{"type": "image", "image": imgs}]
                content.append({"type": "text", "text": conversation["value"]})
                prompt = [{"role": "user", "content": content}]
                prompts.append(prompt)
                
                # 使用 answer_template 作为 completion（后续用 VRT 替换 <|Obj_x|>）
                completion_raw = answers_data.get("answer_template", "")
                
                # 从 segmentation 数据中提取物体信息
                objects_info = {}
                if "seg" in example and len(example["seg"]) > 0:
                    # 获取第一帧的 segmentation 数据
                    seg_data = example["seg"][0]
                    
                    # 解析 segmentation 数据（如果是字符串需要解析）
                    if isinstance(seg_data, str):
                        seg_data = json.loads(seg_data)
                    
                    # 从 completion 中提取引用的物体编号
                    pattern = r'<\|Obj_(\d+)\|>'
                    obj_indices = re.findall(pattern, completion_raw)
                    
                    # 为每个引用的物体收集来自不同视角的信息
                    for obj_idx in obj_indices:
                        if obj_idx not in objects_info:
                            obj_info_list = []
                            
                            # 检查每个视角的 segmentation 数据
                            for seg_key, seg_value in seg_data.items():
                                if isinstance(seg_value, str):
                                    seg_value = json.loads(seg_value)
                                
                                # 如果该视角中有这个物体
                                if obj_idx in seg_value:
                                    obj_data = seg_value[obj_idx]
                                    obj_info = {
                                        'bbox': obj_data.get('bbox', []),
                                        'patches': obj_data.get('patches', []),
                                        'label': obj_data.get('label', ''),
                                        'view': seg_key  # 标记来自哪个视角
                                    }
                                    if 'mask' in obj_data:
                                        obj_info['rle'] = obj_data['mask']
                                    obj_info_list.append(obj_info)
                                    # 收集 patch id 用于与当前 grid 对齐的 sanity check
                                    try:
                                        patch_ids_all.extend([int(p) for p in obj_info['patches']])
                                    except Exception:
                                        pass
                            
                            # 如果找到了该物体的信息（可能来自1-2个视角）
                            if obj_info_list:
                                objects_info[obj_idx] = obj_info_list
                
                # 构建 solution 数据结构
                # 将 <|Obj_x|> 替换为 VRT token（使用 patches 信息）
                def _obj_to_vrt(match: re.Match) -> str:
                    obj_idx = match.group(1)
                    view_infos = objects_info.get(obj_idx, [])
                    # 保留有 patch 的视角
                    view_infos = [v for v in view_infos if v.get('patches')]
                    if not view_infos:
                        return match.group(0)

                    # 视角优先级：agent/third/ego -> wrist/hand -> 其他
                    ordered: list[dict] = []
                    def _append_by_substring(subs: list[str]):
                        for sub in subs:
                            for v in view_infos:
                                if sub in v.get('view', '') and v not in ordered:
                                    ordered.append(v)
                    _append_by_substring(["agent", "third", "ego"])
                    _append_by_substring(["wrist", "hand"])
                    for v in view_infos:
                        if v not in ordered:
                            ordered.append(v)

                    picked: list[int] = []
                    sample_n = 3
                    for v in ordered:
                        patches = [int(p) for p in v.get('patches', []) or []]
                        if not patches:
                            continue
                        if len(patches) < sample_n:
                            picked.extend([random.choice(patches) for _ in range(sample_n)])
                        else:
                            picked.extend(random.sample(patches, sample_n))

                    if not picked:
                        return match.group(0)
                    return processor.pid2vrt(picked)

                # 暂存原始 completion 与 objects_info，稍后重映射 patches 后再做 VRT 替换
                completion_raw_list.append(completion_raw)
                objects_infos_list.append(objects_info)
            else:
                # 如果没有对话数据，跳过
                return None
        
        # 使用 padt_vl_interface 的 tokenizer 处理
        prompt_texts = [self.padt_vl_interface.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        ) for prompt in prompts]
        
        # 使用所有视角的图片（和 build_padtvl_inputs 一致，保留多视角信息）
        from qwen_vl_utils import process_vision_info
        
        # prompts 已经包含 resize 后的图片信息，直接用于 process_vision_info
        image_inputs, _ = process_vision_info(prompts)
        
        # Tokenize prompts with images
        prompt_inputs = self.padt_vl_interface.processor(
            text=prompt_texts,
            images=image_inputs,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )

        prompt_inputs = {k: v.to(self.padt_vl_interface.model.device) for k, v in prompt_inputs.items()}
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        
        # --- Patch remapping: align offline patches (computed at long-side 644, patch 28) to runtime grid ---
        def _remap_patches(patches: list[int], off_h: int, off_w: int, rt_h: int, rt_w: int) -> list[int]:
            if not patches or off_h <= 0 or off_w <= 0 or rt_h <= 0 or rt_w <= 0:
                return patches
            mapped: list[int] = []
            for p in patches:
                r_off = p // off_w
                c_off = p % off_w
                r_norm = (r_off + 0.5) / off_h
                c_norm = (c_off + 0.5) / off_w
                r_rt = int(torch.floor(torch.tensor(r_norm * rt_h)).item())
                c_rt = int(torch.floor(torch.tensor(c_norm * rt_w)).item())
                r_rt = max(0, min(rt_h - 1, r_rt))
                c_rt = max(0, min(rt_w - 1, c_rt))
                mapped.append(r_rt * rt_w + c_rt)
            # 去重且保持稳定顺序
            seen = set()
            uniq = []
            for m in mapped:
                if m not in seen:
                    seen.add(m)
                    uniq.append(m)
            return uniq

        # 预计算每个 sample 的离线/在线网格尺寸（用每个样本的第一张图，假设多视角同分辨率）
        grid_thw = prompt_inputs.get('image_grid_thw')
        runtime_grid_hw = []  # [(rt_h, rt_w)] per sample
        offline_grid_hw = []  # [(off_h, off_w)] per sample
        if grid_thw is not None:
            flat_orig_sizes = []
            for imgs in batch_images:
                if isinstance(imgs, (list, tuple)):
                    flat_orig_sizes.extend([img.size for img in imgs])
                else:
                    flat_orig_sizes.append(imgs.size)
            flat_idx = 0
            for imgs in batch_images:
                count = len(imgs) if isinstance(imgs, (list, tuple)) else 1
                # runtime grid 用第一视角
                if flat_idx < grid_thw.shape[0]:
                    rt_h = int(grid_thw[flat_idx, 1].item())
                    rt_w = int(grid_thw[flat_idx, 2].item())
                else:
                    rt_h = rt_w = 0
                # offline grid 用第一视角原始尺寸
                if flat_idx < len(flat_orig_sizes):
                    ow, oh = flat_orig_sizes[flat_idx]
                    scale = 644.0 / max(oh, ow)
                    target_w = int(ow * scale)
                    target_h = int(oh * scale)
                    off_w = max(1, round(target_w / 28))
                    off_h = max(1, round(target_h / 28))
                else:
                    off_h = off_w = 0
                runtime_grid_hw.append((rt_h, rt_w))
                offline_grid_hw.append((off_h, off_w))
                flat_idx += count

        # 按样本重映射 patches，并重新生成 completions（VRT token 与 runtime grid 对齐）
        completions = []
        solutions = []
        if grid_thw is not None and runtime_grid_hw:
            for sample_idx, (rt_hw, off_hw) in enumerate(zip(runtime_grid_hw, offline_grid_hw)):
                if sample_idx >= len(objects_infos_list) or sample_idx >= len(completion_raw_list):
                    break
                rt_h, rt_w = rt_hw
                off_h, off_w = off_hw
                objects_info = objects_infos_list[sample_idx]
                for obj_idx, obj_info_list in objects_info.items():
                    for obj_info in obj_info_list:
                        if 'patches' in obj_info:
                            obj_info['patches'] = _remap_patches(obj_info['patches'], off_h, off_w, rt_h, rt_w)
                completion_raw = completion_raw_list[sample_idx]

                def _obj_to_vrt_runtime(match: re.Match) -> str:
                    obj_idx = match.group(1)
                    view_infos = objects_info.get(obj_idx, [])
                    view_infos = [v for v in view_infos if v.get('patches')]
                    if not view_infos:
                        return match.group(0)
                    ordered: list[dict] = []
                    def _append_by_substring(subs: list[str]):
                        for sub in subs:
                            for v in view_infos:
                                if sub in v.get('view', '') and v not in ordered:
                                    ordered.append(v)
                    _append_by_substring(["agent", "third", "ego"])
                    _append_by_substring(["wrist", "hand"])
                    for v in view_infos:
                        if v not in ordered:
                            ordered.append(v)
                    picked: list[int] = []
                    sample_n = 3
                    for v in ordered:
                        patches = [int(p) for p in v.get('patches', []) or []]
                        if not patches:
                            continue
                        if len(patches) < sample_n:
                            picked.extend([random.choice(patches) for _ in range(sample_n)])
                        else:
                            picked.extend(random.sample(patches, sample_n))
                    if not picked:
                        return match.group(0)
                    return processor.pid2vrt(picked)

                completion = re.sub(r'<\|Obj_(\d+)\|>', _obj_to_vrt_runtime, completion_raw)
                completions.append(completion)
                solutions.append({"text": completion, "objects": objects_info})
        else:
            # 回退：若无 grid 信息则直接使用原始 completion_raw（不推荐）
            for completion_raw, objects_info in zip(completion_raw_list, objects_infos_list):
                completions.append(completion_raw)
                solutions.append({"text": completion_raw, "objects": objects_info})

        # Tokenize completions（已对齐 runtime grid 的 VRT）
        completion_inputs = self.padt_vl_interface.processor(
            text=completions,
            return_tensors='pt',
            padding=True,
            padding_side='right',
            add_special_tokens=False
        )
        completion_inputs = {k: v.to(self.padt_vl_interface.model.device) for k, v in completion_inputs.items()}
        completion_ids = completion_inputs["input_ids"]
        completion_mask = completion_inputs["attention_mask"]

        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # 仅首次打印一次与 patches 对齐相关的关键信息，方便排查 VLM loss 是否错位
        if not hasattr(self, "_logged_vlm_patch_debug") or not self._logged_vlm_patch_debug:
            grid_info = grid_thw.detach().cpu().tolist() if grid_thw is not None else None
            total_patches = None
            if grid_thw is not None and grid_thw.numel() >= 3:
                # grid_thw: [B, 3] -> t, h, w；实际视觉网格大小为 h*w
                h = int(grid_thw[0, 1].item())
                w = int(grid_thw[0, 2].item())
                total_patches = h * w
            min_patch = min(patch_ids_all) if patch_ids_all else None
            max_patch = max(patch_ids_all) if patch_ids_all else None
            logger.warning(
                "[VLM loss debug] vlm_image_size=%s, image_grid_thw=%s, total_patches=%s, patch_id_min=%s, patch_id_max=%s, images_per_sample=%s, rt_hw_sample0=%s, off_hw_sample0=%s",
                vlm_image_size,
                grid_info,
                total_patches,
                min_patch,
                max_patch,
                images_per_sample,
                runtime_grid_hw[0] if runtime_grid_hw else None,
                offline_grid_hw[0] if offline_grid_hw else None,
            )
            self._logged_vlm_patch_debug = True

        # 重新映射 VRT token 到全局 ID（与 PaDT 训练对齐）
        if prompt_inputs.get('image_grid_thw') is not None:
            input_ids = processor.assign_to_global_vrt_id(input_ids, prompt_inputs['image_grid_thw'], images_per_sample=images_per_sample)
        
        # Prepare multimodal inputs
        multimodal_inputs = {
            'image_grid_thw': prompt_inputs.get('image_grid_thw'),
            'pixel_values': prompt_inputs.get('pixel_values')
        }
        
        # Forward pass with gradients enabled for VLM
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_output = self.padt_vl_interface.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **multimodal_inputs
            )
            logits = model_output.logits[:, prompt_length-1:-1, :]  # (B, L, V)
        
        # Compute token loss（在 no_grad 外计算，但 logits 已 detach，不回传 VLM 梯度）
        target_ids = input_ids[:, prompt_length:]  # (B, L)
        
        # Calculate cross-entropy loss
        logit_log_probs = F.log_softmax(logits, dim=-1)
        token_log_prob = torch.gather(logit_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        per_token_loss = -token_log_prob
        
        # Average over valid tokens
        vlm_loss = ((per_token_loss * completion_mask).sum(dim=-1) / (completion_mask.sum(dim=-1) + 1e-4)).mean()
        
        return vlm_loss

    def forward(
        self,
        examples: List[dict] = None,
        compute_vlm_loss: bool = False,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
                - solution: str (optional, for VLM training)
            compute_vlm_loss: bool, whether to compute VLM loss
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
                vlm_loss (torch.Tensor, optional): VLM language modeling loss.
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example.get("lang", example.get("language")) for example in examples]  # [B, str]
        # print(f"instruction: {instructions[0]}")
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        # Step 1: QWenVL input format (for action prediction, no solutions needed)
        padt_inputs = self.padt_vl_interface.build_padtvl_inputs(
            images=batch_images, 
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            padt_outputs = self.padt_vl_interface(
                **padt_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 取与 DiT 层数匹配的最后 N 层隐藏态，按层喂给 DiT
            all_hidden = padt_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            # detach VLM features for action head: action loss only trains DiT,
            # VLM loss only trains VLM — prevents gradient conflict
            detach_action_features = getattr(self.config.trainer, "detach_action_features", False) if self.config and self.config.trainer else False
            if detach_action_features:
                vl_embs_list = [h.detach() for h in all_hidden[-expected_layers:]]
            else:
                vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            repeated_diffusion_steps = 2 # NO repeat for big action FM (use config value instead)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # 对每层特征做 repeat
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        # 构建返回字典
        output_dict = {"action_loss": action_loss}
        
        # 计算 VLM loss（如果需要且数据可用）
        if compute_vlm_loss:
            # 释放 action path 的中间变量，腾出 GPU 显存给 VLM loss forward
            del padt_inputs, padt_outputs, all_hidden, vl_embs_list, base_hidden
            del vl_embs_list_repeated, actions_target_repeated, actions_target
            if state_repeated is not None:
                del state_repeated
            torch.cuda.empty_cache()
            
            vlm_loss = self.compute_vlm_loss(examples, batch_images, instructions)
            if vlm_loss is not None:
                output_dict["vlm_loss"] = vlm_loss
        
        return output_dict



    @torch.inference_mode()
    def predict_action( # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        from deployment.model_server.tools.image_tools import to_pil_preserve
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PIL]]
        instructions = [example.get("lang", example.get("language")) for example in examples]  # [B, str]
        if any(instr is None for instr in instructions):
            missing = [i for i, instr in enumerate(instructions) if instr is None]
            raise KeyError(f"Missing instruction key ('lang' or 'language') in examples at indices {missing}")
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: PaDT VL input format (use padt_vl_interface)
        padt_inputs = self.padt_vl_interface.build_padtvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            padtvl_outputs = self.padt_vl_interface(
                **padt_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = padtvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    

    model = PaDT_PI(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])

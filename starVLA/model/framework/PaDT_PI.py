# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
import re
import os
import cv2
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple,Union,Sized
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch.distributed as dist



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.common_utils import get_input_embedding_vocab_size, resolve_llm_hidden_size
from starVLA.model.modules.vlm import get_vlm_model
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
        qwenvl_cfg = (
            self.config.framework.qwenvl
            if hasattr(self.config, "framework") and hasattr(self.config.framework, "qwenvl")
            else None
        )
        strict_vrt_init = bool(qwenvl_cfg.get("strict_vrt_init", True)) if qwenvl_cfg is not None else True
        # Pre-reserve VRT tokens and resize embeddings once BEFORE deepspeed/accelerate wraps the model
        try:
            merge_size = int(getattr(self.padt_vl_interface.processor, "spatial_merge_size", 1))
            if merge_size < 1:
                raise ValueError(f"Invalid spatial_merge_size={merge_size}, expected >= 1")

            max_vrt_patches = qwenvl_cfg.get("max_vrt_patches", None) if qwenvl_cfg is not None else None
            if max_vrt_patches is None:
                # conservative default: 4096 visual patches per sample
                max_vrt_patches = 4096
            max_vrt_patches = int(max_vrt_patches)

            # Minimum required VRT patches for one sample (default assumes 2 views for PaDTPI).
            image_size_cfg = getattr(getattr(getattr(self.config, "datasets", None), "vla_data", None), "image_size", 224)
            if isinstance(image_size_cfg, (list, tuple)):
                image_size = int(image_size_cfg[0])
            else:
                image_size = int(image_size_cfg)
            patches_per_side = max(1, image_size // 14)
            per_image_vrt_patches = max(1, (patches_per_side * patches_per_side) // (merge_size ** 2))
            images_per_sample = int(qwenvl_cfg.get("images_per_sample", 2)) if qwenvl_cfg is not None else 2
            min_required_vrt_patches = (
                int(qwenvl_cfg.get("min_required_vrt_patches", per_image_vrt_patches * images_per_sample))
                if qwenvl_cfg is not None
                else per_image_vrt_patches * images_per_sample
            )
            if max_vrt_patches < min_required_vrt_patches:
                msg = (
                    f"max_vrt_patches={max_vrt_patches} is smaller than minimum required "
                    f"{min_required_vrt_patches} (image_size={image_size}, merge_size={merge_size}, "
                    f"images_per_sample={images_per_sample})"
                )
                if strict_vrt_init:
                    raise ValueError(msg)
                logger.warning(msg + "; auto-upgrade max_vrt_patches to minimum required.")
                max_vrt_patches = min_required_vrt_patches

            # grid_thw so that t*h*w / merge_size^2 == max_vrt_patches
            grid_thw = torch.tensor([[1, 1, max_vrt_patches * (merge_size ** 2)]], dtype=torch.int64)
            # add tokens to tokenizer if needed
            self.padt_vl_interface.processor.set_image_grid_thw(grid_thw)
            tok_len = len(self.padt_vl_interface.processor.tokenizer)
            embed_len = get_input_embedding_vocab_size(self.padt_vl_interface.model)
            if tok_len > embed_len:
                logger.info(
                    "Tokenizer has %s extra VRT tokens above base embeddings (%s -> %s); "
                    "skip resize_token_embeddings on purpose (VRT ids map to runtime image prototypes).",
                    tok_len - embed_len,
                    embed_len,
                    tok_len,
                )
            # Keep base embedding vocab size (not tokenizer length after adding VRT tokens).
            self.padt_vl_interface.processor.model_embed_token_size = embed_len
        except Exception as e:
            msg = f"VRT pre-resize failed: {e}"
            if strict_vrt_init:
                raise RuntimeError(msg) from e
            logger.warning(msg + "; training may resize at runtime.")
        

        # dynamic get llm config (Qwen2.5 VL configs sometimes omit top-level hidden_size)
        llm_hidden_size = resolve_llm_hidden_size(self.padt_vl_interface.model)

        vl_config = self.padt_vl_interface.model.config
        num_vl_layers = getattr(vl_config, "num_hidden_layers", None)
        if num_vl_layers is None and hasattr(vl_config, "text_config"):
            num_vl_layers = getattr(vl_config.text_config, "num_hidden_layers", None)
        if num_vl_layers is None:
            model_core = getattr(self.padt_vl_interface.model, "model", None)
            layers = getattr(model_core, "layers", None)
            if layers is not None:
                num_vl_layers = len(layers)
        if num_vl_layers is None:
            raise AttributeError("Cannot resolve num_vl_layers from Qwen2_5_VL config/model")
        num_vl_layers = int(num_vl_layers)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        trainer_cfg = getattr(self.config, "trainer", None)
        self.debug_vrt = bool(getattr(trainer_cfg, "debug_vrt", False))
        self.debug_vrt_max_steps = int(getattr(trainer_cfg, "debug_vrt_max_steps", 20))
        self.debug_vrt_token_preview = int(getattr(trainer_cfg, "debug_vrt_token_preview", 12))
        self.debug_vrt_visualize = bool(getattr(trainer_cfg, "debug_vrt_visualize", self.debug_vrt))
        self.debug_vrt_visualize_max_samples = int(getattr(trainer_cfg, "debug_vrt_visualize_max_samples", 2))
        self._debug_vrt_step = 0
        self._debug_vrt_vis_count = 0
        output_dir = str(getattr(self.config, "output_dir", "."))
        self.debug_vrt_vis_dir = str(
            getattr(trainer_cfg, "debug_vrt_visualize_dir", os.path.join(output_dir, "vrt_debug_vis"))
        )
        if self.debug_vrt_visualize and self._is_main_process():
            os.makedirs(self.debug_vrt_vis_dir, exist_ok=True)
        self.padt_vl_interface.processor.debug_vrt = self.debug_vrt
        self.padt_vl_interface.processor.debug_vrt_max_calls = self.debug_vrt_max_steps
        self.padt_vl_interface.processor.debug_vrt_token_preview = self.debug_vrt_token_preview

    def _is_main_process(self) -> bool:
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def _vrt_debug_this_step(self) -> bool:
        return self.debug_vrt and self._is_main_process() and self._debug_vrt_step < self.debug_vrt_max_steps

    def _vrt_log(self, message: str) -> None:
        logger.info(f"[VRTDBG] {message}")

    @staticmethod
    def _patch_box_from_pid(pid: int, patch_w: int, patch_h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        patch_x = int(pid % patch_w)
        patch_y = int(pid // patch_w)
        px_w = img_w / float(patch_w)
        px_h = img_h / float(patch_h)
        x0 = int(round(patch_x * px_w))
        y0 = int(round(patch_y * px_h))
        x1 = int(round((patch_x + 1) * px_w))
        y1 = int(round((patch_y + 1) * px_h))
        return x0, y0, x1, y1

    def _save_vrt_patch_visualization(
        self,
        sample_images: List[PIL.Image.Image],
        sample_records: List[dict],
        patch_w: int,
        patch_h: int,
        debug_step: int,
        sample_idx: int,
    ) -> Optional[str]:
        if not self.debug_vrt_visualize or not self._is_main_process():
            return None
        if len(sample_images) == 0:
            return None

        # Use a fixed color palette for object-level consistency inside one sample.
        colors = [
            (255, 80, 80),
            (80, 180, 255),
            (80, 220, 120),
            (255, 190, 80),
            (200, 120, 255),
            (255, 120, 200),
        ]

        vis_images: List[PIL.Image.Image] = []
        for view_id, img in enumerate(sample_images):
            vis = to_pil_preserve(img).copy().convert("RGB")
            draw = PIL.ImageDraw.Draw(vis, "RGBA")
            for rec_idx, rec in enumerate(sample_records):
                color = colors[rec_idx % len(colors)]
                label_text = rec.get("label", "") or rec.get("obj_token", "obj")

                # Draw all candidate patches in this view (thin line).
                for pid, vid in zip(rec.get("candidate_local", []), rec.get("candidate_view_ids", [])):
                    if int(vid) != view_id:
                        continue
                    x0, y0, x1, y1 = self._patch_box_from_pid(int(pid), patch_w, patch_h, vis.width, vis.height)
                    draw.rectangle((x0, y0, x1, y1), outline=(*color, 180), width=1)

                # Draw selected patches in this view (thick line + text).
                for pid, vid in zip(rec.get("picked_local", []), rec.get("picked_view_ids", [])):
                    if int(vid) != view_id:
                        continue
                    x0, y0, x1, y1 = self._patch_box_from_pid(int(pid), patch_w, patch_h, vis.width, vis.height)
                    draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=3)
                    text_bg = (x0, max(0, y0 - 14), min(vis.width, x0 + 140), y0)
                    draw.rectangle(text_bg, fill=(0, 0, 0, 150))
                    draw.text((x0 + 2, max(0, y0 - 13)), label_text, fill=(255, 255, 255, 255))

            # View title.
            view_name = "agentview" if view_id == 0 else "wrist"
            draw.rectangle((0, 0, 160, 16), fill=(0, 0, 0, 180))
            draw.text((4, 2), view_name, fill=(255, 255, 255, 255))
            vis_images.append(vis)

        canvas_w = sum(im.width for im in vis_images)
        canvas_h = max(im.height for im in vis_images)
        canvas = PIL.Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
        x_offset = 0
        for im in vis_images:
            canvas.paste(im, (x_offset, 0))
            x_offset += im.width

        save_path = os.path.join(
            self.debug_vrt_vis_dir,
            f"step_{debug_step:05d}_sample_{sample_idx:02d}.png",
        )
        canvas.save(save_path)
        return save_path

    def compute_vlm_loss(self, model, examples: List[dict], batch_images :List[List[PIL.Image.Image]], padt_vlm_inputs: List[dict]) -> Optional[torch.Tensor]:
        
        prompt_text = padt_vlm_inputs
        debug_this_step = self._vrt_debug_this_step()
        if debug_this_step:
            self._debug_vrt_step += 1
            self._vrt_log(
                f"compute_vlm_loss step={self._debug_vrt_step} batch_size={len(examples)} "
                f"prompt_example={repr(prompt_text[0][:200] if len(prompt_text) > 0 else '')}"
            )

        completions = []
        solutions = []
        preview_n = max(1, int(self.debug_vrt_token_preview))

        for idx, x in enumerate(examples):
            image_1 = batch_images[idx][0]
            image_2 = batch_images[idx][1] if len(batch_images[idx]) > 1 else None
            if debug_this_step and idx < 2:
                self._vrt_log(
                    f"sample={idx} image_sizes="
                    f"{getattr(image_1, 'size', None)}, {getattr(image_2, 'size', None)}"
                )

            # completion
            im_w, im_h = batch_images[idx][0].size
            patch_w, patch_h = round(im_w / 14), round(im_h / 14)

            solution = x.get('answers', {})
            completion = solution.get('answer_template', "")
            pattern = r'(<\|Obj_(\d+)\|>)'
            obj_in_completion = re.findall(pattern, completion)
            obj_strs = [i[0] for i in obj_in_completion]
            if debug_this_step and idx == 0:
                self._vrt_log(
                    f"sample=0 obj_in_completion={obj_strs} answer_template={repr(completion[:200])}"
                )

            seg = x.get('seg', {})
            if isinstance(seg, list):
                seg = seg[0] if len(seg) > 0 else {}
            seg = seg if isinstance(seg, dict) else {}
            agentview_seg = seg.get('agentview_bbox_mask', {})
            wrist_seg = seg.get('wrist_bbox_mask', {})
            agent_objs = [agentview_seg.get(str(i[1])) for i in obj_in_completion]
            wrist_objs = [wrist_seg.get(str(i[1])) for i in obj_in_completion]
            merged_objs = []
            sample_vis_records = []
            for agent_obj, wrist_obj in zip(agent_objs, wrist_objs):
                if not isinstance(agent_obj, dict) and not isinstance(wrist_obj, dict):
                    merged_objs.append(None)
                    continue

                obj_merged = {}
                merged_patches = []
                merged_patch_view_ids = []
                if isinstance(agent_obj, dict):
                    obj_merged.update(agent_obj)
                    if "label" not in obj_merged:
                        obj_merged["label"] = (
                            agent_obj.get("label")
                            or (agent_obj.get("mask", {}) or {}).get("label", "")
                        )
                    agent_patches = [int(p) for p in (agent_obj.get('patches', []) or [])]
                    merged_patches.extend(agent_patches)
                    merged_patch_view_ids.extend([0] * len(agent_patches))  # 0 -> agent (image[0])
                if isinstance(wrist_obj, dict):
                    if not obj_merged:
                        obj_merged.update(wrist_obj)
                    if not obj_merged.get("label"):
                        obj_merged["label"] = (
                            wrist_obj.get("label")
                            or (wrist_obj.get("mask", {}) or {}).get("label", "")
                        )
                    wrist_patches = [int(p) for p in (wrist_obj.get('patches', []) or [])]
                    merged_patches.extend(wrist_patches)
                    merged_patch_view_ids.extend([1] * len(wrist_patches))  # 1 -> wrist (image[1])

                if len(merged_patches) == 0:
                    merged_objs.append(None)
                    continue
                #  "patches": [agent_p1, agent_p2, ..., wrist_p1, wrist_p2, ...],
                obj_merged['patches'] = merged_patches
                #  "patch_view_ids": [0, 0, ..., 1, 1, ...]
                obj_merged['patch_view_ids'] = merged_patch_view_ids            
                merged_objs.append(obj_merged)

            pattern_without_matching = r'<\|Obj_\d+\|>'
            completion_parts = re.split(pattern_without_matching, completion)
            
            completion_with_vrt = completion_parts[0]
            new_objs = []
            for obj_str, completion_part, obj in zip(obj_strs, completion_parts[1:], merged_objs):
                if obj is None:
                    completion_with_vrt += completion_part
                    continue
                obj_ = obj.copy()
                selected_patches = np.array(obj_['patches'])
                selected_patch_view_ids = np.array(
                    obj_.get('patch_view_ids', [0] * len(selected_patches))
                )
                if selected_patch_view_ids.shape[0] != selected_patches.shape[0]: # 检查存的patch和view_id数量是否一致
                    selected_patch_view_ids = np.zeros_like(selected_patches)
                # sample-local global patch ids across multi-view images:
                # view 0 uses [0..P-1], view 1 uses [P..2P-1], ...视角偏移
                per_view_patch_num = patch_w * patch_h
                selected_patches_global = selected_patches + selected_patch_view_ids * per_view_patch_num
                if debug_this_step and idx == 0:
                    self._vrt_log(
                        f"obj={obj_str} local_patches={selected_patches[:preview_n].tolist()} "
                        f"view_ids={selected_patch_view_ids[:preview_n].tolist()} "
                        f"global_patches={selected_patches_global[:preview_n].tolist()}"
                    )

                # Per-view center selection:
                # choose one center patch for each available view (agent/wrist).
                pick_idx_list = []
                # 每个视角单独处理
                for view_id in sorted(np.unique(selected_patch_view_ids).tolist()):
                    view_mask = selected_patch_view_ids == view_id
                    view_indices = np.where(view_mask)[0]
                    if view_indices.size == 0:
                        continue

                    view_patches = selected_patches[view_indices]
                    view_x = view_patches % patch_w
                    view_y = view_patches // patch_w
                    left_m = view_x == view_x.min()
                    right_m = view_x == view_x.max()
                    top_m = view_y == view_y.min()
                    bottom_m = view_y == view_y.max()
                    centre_m = (left_m + right_m + top_m + bottom_m) == 0

                    centre_local = np.where(centre_m)[0]
                    if centre_local.size == 0:
                        centre_local = np.arange(view_indices.size)

                    chosen_local = np.random.choice(centre_local)
                    pick_idx_list.append(view_indices[chosen_local])


                pick_idx = np.array(pick_idx_list, dtype=np.int64)
                pick_patch_local = selected_patches[pick_idx]
                pick_patch = selected_patches_global[pick_idx]
            
                obj_['patches'] = selected_patches_global
                obj_['picked'] = pick_patch
                obj_['picked_local'] = pick_patch_local
                obj_['picked_view_ids'] = selected_patch_view_ids[pick_idx]
                new_objs.append(obj_)
                if debug_this_step and idx < self.debug_vrt_visualize_max_samples:
                    sample_vis_records.append(
                        {
                            "obj_token": obj_str,
                            "label": obj_.get("label", obj_str),
                            "candidate_local": selected_patches.tolist(),
                            "candidate_view_ids": selected_patch_view_ids.tolist(),
                            "picked_local": pick_patch_local.tolist(),
                            "picked_view_ids": selected_patch_view_ids[pick_idx].tolist(),
                        }
                    )
                if debug_this_step and idx == 0:
                    self._vrt_log(
                        f"obj={obj_str} picked_local={pick_patch_local.tolist()} "
                        f"picked_global={pick_patch.tolist()} picked_view_ids={selected_patch_view_ids[pick_idx].tolist()}"
                    )
                completion_with_vrt += self.padt_vl_interface.processor.pid2vrt(pick_patch) + completion_part

            solutions.append({
                'text': solution.get('text', completion),
                'objects': new_objs
            })
            completions.append(completion_with_vrt + self.padt_vl_interface.processor.tokenizer.eos_token)
            if debug_this_step and idx < self.debug_vrt_visualize_max_samples and len(sample_vis_records) > 0:
                vis_path = self._save_vrt_patch_visualization(
                    sample_images=batch_images[idx],
                    sample_records=sample_vis_records,
                    patch_w=patch_w,
                    patch_h=patch_h,
                    debug_step=self._debug_vrt_step,
                    sample_idx=idx,
                )
                if vis_path is not None:
                    self._vrt_log(f"saved_patch_vis={vis_path}")

        if debug_this_step and len(completions) > 0:
            self._vrt_log(
                f"completion_with_vrt_example={repr(completions[0][:300])} "
                f"vrt_count_in_completion={completions[0].count('<|VRT_')}"
            )

        # tokenizing
        prompt_inputs = self.padt_vl_interface.processor(
            text=prompt_text,
            images=batch_images,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )
        model_device = next(model.parameters()).device
        prompt_inputs = prompt_inputs.to(model_device)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        batch_size = prompt_ids.size(0)
        prompt_length = prompt_ids.size(1)
        multimodal_inputs = {
            'image_grid_thw': prompt_inputs['image_grid_thw'],
            'pixel_values': prompt_inputs['pixel_values']
        }
        # trainer_cfg = getattr(self.config, "trainer", None)
        # use_sft_vp_mask = bool(getattr(trainer_cfg, "use_sft_vp_mask", False))
        use_sft_vp_mask = True
        model_embed_token_size = int(
            getattr(self.padt_vl_interface.processor, "model_embed_token_size", get_input_embedding_vocab_size(model))
        )
        device = prompt_ids.device
        images_per_sample = [
            len(imgs) if isinstance(imgs, (list, tuple)) else 1
            for imgs in batch_images
        ]
        if debug_this_step:
            self._vrt_log(
                f"tokenize prompt_length={prompt_length} model_embed_token_size={model_embed_token_size} "
                f"images_per_sample={images_per_sample} image_grid_thw={multimodal_inputs['image_grid_thw'].tolist()}"
            )
        
        completion_inputs = self.padt_vl_interface.processor(
            text=completions,
            return_tensors='pt',
            padding=True,
            padding_side='right',
            add_special_tokens=False
        )
        completion_inputs = completion_inputs.to(model_device)
        completion_ids, completion_mask = completion_inputs["input_ids"], completion_inputs["attention_mask"]

        # prepare for Robust Per-token Cross-Entropy Loss.
        loss_masks = []
        gt_bboxes = []
        per_image_patch_nums = multimodal_inputs['image_grid_thw'].cumprod(-1)[:, -1] // (
            self.padt_vl_interface.model.config.vision_config.spatial_merge_size ** 2
        )
        sample_patch_nums = []
        img_ptr = 0
        for n_imgs in images_per_sample:
            sample_patch_nums.append(per_image_patch_nums[img_ptr:img_ptr + n_imgs].sum())
            img_ptr += n_imgs
        sample_patch_nums = torch.stack(sample_patch_nums)
        sample_patch_offsets = torch.nn.functional.pad(sample_patch_nums.cumsum(-1), (1, 0), 'constant', 0)[:-1]
        all_vision_patch_nums = int(sample_patch_nums.sum().item())
        if debug_this_step:
            self._vrt_log(
                f"per_image_patch_nums={per_image_patch_nums.tolist()} sample_patch_nums={sample_patch_nums.tolist()} "
                f"sample_patch_offsets={sample_patch_offsets.tolist()} all_vision_patch_nums={all_vision_patch_nums}"
            )

        for sol, vpn in zip(solutions, sample_patch_offsets):
            vpn_i = int(vpn.item())
            for obj in sol['objects']:
                this_object_loss_mask = torch.zeros((obj['picked'].shape[0], all_vision_patch_nums), device=device, dtype=torch.bool)
                this_object_loss_mask[:, vpn_i + np.array(obj['patches'])] = True
                this_object_loss_mask[np.arange(obj['picked'].shape[0]), vpn_i + obj['picked']] = False
                loss_masks.append(this_object_loss_mask)
                # obj['bbox']: x1, y1, x2, y2. Value in [0, 1].
                gt_bboxes.append(obj['bbox'])

        if len(loss_masks) == 0:
            if debug_this_step:
                self._vrt_log("no valid objects for VLM loss in this batch -> return None")
            return None

        # Keep VP-only mask first; full-vocab padding will be aligned to runtime logits size later.
        loss_masks = torch.cat(loss_masks, dim=0)
        gt_bboxes = torch.Tensor(gt_bboxes).to(device).to(torch.bfloat16)
        if len(gt_bboxes.shape) == 1:
            gt_bboxes = gt_bboxes.unsqueeze(dim=-1).repeat_interleave(4, dim=-1)

        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pre_global_ids = input_ids.clone() if debug_this_step else None

        input_ids = self.padt_vl_interface.processor.assign_to_global_vrt_id(
            input_ids,
            multimodal_inputs['image_grid_thw'],
            images_per_sample=images_per_sample,
        )
        if debug_this_step:
            pre_mask = pre_global_ids >= model_embed_token_size
            post_mask = input_ids >= model_embed_token_size
            pre_vrt = pre_global_ids[pre_mask][:preview_n].tolist()
            post_vrt = input_ids[post_mask][:preview_n].tolist()
            self._vrt_log(
                f"assign_to_global_vrt_id pre_vrt_head={pre_vrt} post_vrt_head={post_vrt} "
                f"pre_vrt_count={int(pre_mask.sum().item())} post_vrt_count={int(post_mask.sum().item())}"
            )

        # Get the current policy's log probabilities
        model_output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **multimodal_inputs)

        logits = model_output.logits[:, prompt_length-1:-1, :]  # (B, L, V)
        input_ids = input_ids[:, prompt_length:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        logits_vocab_size = int(logits.shape[-1])
        expected_vocab_size = int(model_embed_token_size + all_vision_patch_nums)
        if logits_vocab_size != expected_vocab_size:
            raise ValueError(
                "VLM vocab alignment mismatch: "
                f"logits_vocab_size={logits_vocab_size}, "
                f"expected(model_embed_token_size + all_vision_patch_nums)={expected_vocab_size}, "
                f"model_embed_token_size={model_embed_token_size}, "
                f"all_vision_patch_nums={all_vision_patch_nums}, "
                f"image_grid_thw={multimodal_inputs['image_grid_thw'].tolist()}, "
                f"images_per_sample={images_per_sample}"
            )

        if use_sft_vp_mask:
            loss_masks = torch.nn.functional.pad(loss_masks, (model_embed_token_size, 0), 'constant', False)
            visual_patch_mask = input_ids >= model_embed_token_size
            logits[visual_patch_mask] = logits[visual_patch_mask].masked_fill(loss_masks, float('-inf'))
            if debug_this_step:
                self._vrt_log(
                    f"use_sft_vp_mask visual_patch_tokens={int(visual_patch_mask.sum().item())} "
                    f"loss_masks_shape={tuple(loss_masks.shape)} logits_shape={tuple(logits.shape)} "
                    f"model_embed_token_size={model_embed_token_size}"
                )
        
        # # decode to bbox
        # hidden_states = torch.stack(model_output.hidden_states, dim=1)[:, -1:, prompt_length-1:-1].permute(2, 1, 0, 3).unsqueeze(dim=-2).contiguous() # [BS, Layers, N, Dim] -> [N, Layers, BS, 1, D]
        # completions, feats, labels, vps, vps_feats = parseVRTintoCompletion(self.padt_vl_interface.processor, completion_ids, hidden_states, torch.tensor([False] * batch_size), model_output.past_image_embeds, multimodal_inputs['image_grid_thw']) # hidden_states: [N, Layers, BS, D]
        # low_res_image_embeds = model_output.past_image_embeds
        # high_res_image_embeds = model_output.past_high_res_image_embeds
        # visual_pe = model_output.past_visual_pe
        
        # # warm up stage: using visual prototype rather than hidden features to feed into decoder.
        # if self.state.epoch < (self.state.num_train_epochs / 4) and self.state.global_step < 300 and self.args.use_warm_up:
        #     feats = vps_feats
        # decoded_list = model(feats, low_res_image_embeds, high_res_image_embeds, multimodal_inputs['image_grid_thw'], visual_pe, is_main=False)
        # del model_output

        # if self.args.use_mask_loss:
        #     gt_mask = torch.zeros_like(decoded_list['pred_mask'])
        #     loss_mask = torch.zeros_like(decoded_list['pred_mask'])

        #     obj_idx = 0
        #     for sol in solutions:
        #         for obj in sol['objects']:
        #             if 'rle' in obj:
        #                 gt_m = mask.decode(obj['rle'])
        #                 mask_h, mask_w = decoded_list['pred_mask_valid_hw'][0][obj_idx].item(), decoded_list['pred_mask_valid_hw'][1][obj_idx].item()
        #                 resized_gt_m = torch.from_numpy(cv2.resize(gt_m.astype(np.float32()), (mask_w * 4, mask_h * 4)) > 0.5).to(gt_mask.dtype).to(gt_mask.device)
        #                 gt_mask[obj_idx, :mask_h * 4, :mask_w * 4] = resized_gt_m
        #                 loss_mask[obj_idx, :mask_h * 4, :mask_w * 4] = 1.0
        #             obj_idx += 1
        #     mask_loss = self.dice_loss(decoded_list['pred_mask'], gt_mask, loss_mask) + self.sigmoid_focal_loss(decoded_list['pred_mask'], gt_mask, loss_mask)
        #     self._metrics['mask_loss'].append(self.accelerator.gather_for_metrics(mask_loss).mean().item())
        # else:
        #     mask_loss = 0.

        # token loss
        logit_log_probs = logits.log_softmax(dim=-1)
        token_log_prob = torch.gather(logit_log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        per_token_loss = -token_log_prob
        sft_loss = ((per_token_loss * completion_mask).sum(dim=-1) / (completion_mask.sum(dim=-1) + 1e-4)).to(logits.dtype)
        if debug_this_step:
            self._vrt_log(
                f"sft_loss_mean={float(sft_loss.detach().mean().item()):.6f} "
                f"completion_valid_tokens={completion_mask.sum(dim=-1).tolist()}"
            )
        if hasattr(self, "_metrics") and isinstance(self._metrics, dict):
            if hasattr(self, "accelerator") and self.accelerator is not None:
                sft_loss_metric = self.accelerator.gather_for_metrics(sft_loss).mean().item()
            else:
                sft_loss_metric = sft_loss.detach().mean().item()
            self._metrics.setdefault('sft_loss', []).append(sft_loss_metric)
        
        # if self.args.use_bbox_loss:
        #     # bbox loss
        #     pred_bboxes = decoded_list['pred_boxes']  # num_bbox, 4 [cx, cy, w, h]
        #     # gt_bboxes # num_bbox, 4 [x1, y1, x2, y2]
        #     num_bboxes = gt_bboxes.shape[0]
        #     giou, iou = self.generalized_box_iou(self.box_cxcywh_to_xyxy(pred_bboxes), gt_bboxes)
        #     giou = torch.diag(giou).to(pred_bboxes.dtype)
        #     bbox_loss = 1. - giou.sum() / (num_bboxes + 1e-4)
        #     bbox_loss += torch.nn.functional.l1_loss(pred_bboxes, self.box_xyxy_to_cxcywh(gt_bboxes), reduction='none').sum() / (num_bboxes + 1e-4)
        #     self._metrics['bbox_loss'].append(self.accelerator.gather_for_metrics(bbox_loss).mean().item())
        #     self._metrics['iou'].append(self.accelerator.gather_for_metrics(torch.diag(iou).sum() / (num_bboxes + 1e-4)).mean().item())
        #     self._metrics['giou'].append(self.accelerator.gather_for_metrics(giou.sum() / (num_bboxes + 1e-4)).mean().item())
        # else:
        #     bbox_loss = 0.

        # if self.args.use_bbox_loss and self.args.use_score_loss:
        #     # score loss
        #     pred_score = decoded_list['pred_score'].sigmoid() * 2. - 1.  # [-1, 1]
        #     score_loss = torch.nn.functional.mse_loss(pred_score, giou.unsqueeze(1).detach(), reduction='sum') / (num_bboxes + 1e-4)
        #     self._metrics['score_loss'].append(self.accelerator.gather_for_metrics(score_loss).mean().item())
        # else:
        #     score_loss = 0.

        #loss = sft_loss.mean() + bbox_loss + score_loss + mask_loss
        return sft_loss.mean()


    def forward(
        self,
        examples: List[dict] = None,
        compute_vlm_loss: bool = False,
        **kwargs,
    ) -> dict:
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
        if compute_vlm_loss and self._vrt_debug_this_step():
            images_per_sample = [len(imgs) if isinstance(imgs, (list, tuple)) else 1 for imgs in batch_images]
            self._vrt_log(
                f"forward batch_size={len(examples)} compute_vlm_loss={compute_vlm_loss} "
                f"images_per_sample={images_per_sample} instruction_example={repr((instructions[0] or '')[:160])}"
            )

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
            vlm_instructions = [
                (ex.get("answers", {}).get("conversations", [{}])[0].get("value", "")
                if ex.get("answers", {}).get("conversations")
                else ex.get("lang", ex.get("language", "")))
                for ex in examples
            ]
            padt_vlm_inputs = self.padt_vl_interface.build_padtvl_vlm_inputs(
                images=batch_images,
                vlm_instructions=vlm_instructions,
            )
            model = self.padt_vl_interface.model
            vlm_loss = self.compute_vlm_loss(model, examples, batch_images, padt_vlm_inputs)
            if vlm_loss is not None: 
                output_dict["vlm_loss"] = vlm_loss
            elif self._vrt_debug_this_step():
                self._vrt_log("compute_vlm_loss returned None for this batch")
        
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

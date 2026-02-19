# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
import transformers
from typing import Optional, List
import copy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoProcessor, AutoConfig
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info
try:
    from ...common_utils import get_input_embedding_vocab_size
except ImportError:
    from starVLA.model.common_utils import get_input_embedding_vocab_size

try:
    from .padt import PaDTForConditionalGeneration
    from .padt_tools import VisonTextProcessingClass
except ImportError:
    from padt import PaDTForConditionalGeneration
    from padt_tools import VisonTextProcessingClass
    
from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# _ACTION_TOKEN_MIN = 151665 # how can we know this range?
# _ACTION_TOKEN_MAX = 153712 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _PaDT_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around PaDT-enhanced Qwen2.5-VL (PaDTForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    Notes:
        - Keeps original model behavior; does not modify internal architecture.
        - Mixed precision handled via torch.autocast in forward / generate.
        - Adaptation layer can be extended for future multi-modal routing if needed.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the PaDT-enhanced Qwen2.5-VL wrapper.

        Parameters:
            config (dict | Any | None):
                Expected to expose a nested attribute/namespace `framework.get("qwenvl", {})`
                where:
                    framework.qwenvl.base_vlm (str): HuggingFace model id or local path.
                Optional expected structure (illustrative):
                    config.framework.get("qwenvl", {}) -> {
                        "base_vlm": "Qwen/Qwen2.5-VL-3B-Instruct"
                    }
                    config.datasets.vla_data.get("CoT_prompt", str) may be used later in build_qwenvl_inputs.
            **kwargs:
                Ignored currently; placeholder for future extension (e.g., override device_map, dtype).

        Side Effects:
            - Downloads / loads pretrained Qwen2.5-VL weights (unless cached).
            - Instantiates AutoProcessor and enforces left padding (required for some FlashAttention paths).

        Attributes Set:
            self.model (PaDTForConditionalGeneration)
            self.processor (AutoProcessor)
            self.config (original config reference)

        Notes:
            - device_map='cuda' is passed to from_pretrained (single or multi-GPU depending on HF accelerate mapping).
            - torch_dtype='auto' lets HF decide best available (prefers bfloat16 on supported hardware).
            - tokenizer padding_side forced to 'left' (important for generation + KV caching alignment).
        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen2.5-VL-3B-Instruct")

        # Force visual merge granularity for VRT/patch alignment (default: 1 => 14x14 @224).
        target_merge_size = int(qwenvl_config.get("spatial_merge_size", 1))
        if target_merge_size < 1:
            raise ValueError(f"Invalid spatial_merge_size={target_merge_size}, expected >= 1")

        # Build model with target merge-size from config-time to avoid runtime inconsistency
        # between precomputed visual indexing buffers and updated merge attributes.
        model_cfg = AutoConfig.from_pretrained(model_id)
        orig_merge_size = int(getattr(model_cfg.vision_config, "spatial_merge_size", 2))
        load_kwargs = dict(
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
        )
        if target_merge_size != orig_merge_size:
            model_cfg.vision_config.spatial_merge_size = target_merge_size
            load_kwargs["config"] = model_cfg
            load_kwargs["ignore_mismatched_sizes"] = True
            logger.warning(
                "Build model with overridden vision spatial_merge_size: %s -> %s (ignore_mismatched_sizes=True)",
                orig_merge_size,
                target_merge_size,
            )

        model = PaDTForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )

        model.config.vision_config.spatial_merge_size = target_merge_size
        updated_modules = 0
        for m in model.modules():
            if hasattr(m, "spatial_merge_size"):
                setattr(m, "spatial_merge_size", target_merge_size)
                updated_modules += 1
            if hasattr(m, "spatial_merge_unit"):
                setattr(m, "spatial_merge_unit", target_merge_size ** 2)
                updated_modules += 1
        if orig_merge_size != target_merge_size:
            logger.warning(
                "Override vision spatial_merge_size: %s -> %s (updated attrs: %s)",
                orig_merge_size,
                target_merge_size,
                updated_modules,
            )

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        # Keep processor-side vision tokenization in sync with model-side spatial merge.
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "merge_size"):
                old_proc_merge = int(getattr(image_processor, "merge_size"))
                if old_proc_merge != target_merge_size:
                    setattr(image_processor, "merge_size", target_merge_size)
                    logger.warning(
                        "Override processor image merge_size: %s -> %s",
                        old_proc_merge,
                        target_merge_size,
                    )
            if hasattr(image_processor, "spatial_merge_size"):
                old_proc_spatial_merge = int(getattr(image_processor, "spatial_merge_size"))
                if old_proc_spatial_merge != target_merge_size:
                    setattr(image_processor, "spatial_merge_size", target_merge_size)
                    logger.warning(
                        "Override processor image spatial_merge_size: %s -> %s",
                        old_proc_spatial_merge,
                        target_merge_size,
                    )

        # Wrap processor to support VRT tokens and pid2vrt helpers
        processor = VisonTextProcessingClass(
            processor,
            spatial_merge_size=target_merge_size,
        )
        # Align tokenizer vocab with model embedding size
        processor.prepare(get_input_embedding_vocab_size(model))

        self.model = model
        self.processor = processor
        self.config = config

        # self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        # self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_image_embeds: Optional[torch.FloatTensor] = None,
        past_logit_mask: Optional[torch.BoolTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying PaDT-enhanced Qwen2.5-VL backbone.

        Args:
            input_ids (LongTensor | None): [B, T] token ids (mutually exclusive with inputs_embeds).
            attention_mask (Tensor | None): [B, T], 1 = attend, 0 = masked.
            position_ids (LongTensor | None): Position indices for RoPE.
            past_key_values (List[FloatTensor] | None): Cached KV states for incremental decoding.
            past_image_embeds (FloatTensor | None): Cached image prototype embeddings from previous forward pass.
            past_logit_mask (BoolTensor | None): Cached logit mask for vocabulary restriction.
            inputs_embeds (FloatTensor | None): [B, T, D] alternative embedding input.
            labels (LongTensor | None): [B, T] LM targets; ignored positions = -100 (IGNORE_INDEX).
            use_cache (bool | None): If True, returns updated past_key_values.
            output_attentions (bool): Whether to include attention maps.
            output_hidden_states (bool): Must be True if downstream modules consume hidden states.
            return_dict (bool): Return HF dataclass if True; else tuple.
            pixel_values (FloatTensor | None): Vision batch (model-specific preprocessed shape).
            pixel_values_videos (FloatTensor | None): Video batch if present.
            image_grid_thw (FloatTensor | None): Optional tiling metadata (e.g., [B, 3] for temporal/height/width splits).
            video_grid_thw (LongTensor | None): Video grid metadata if present.
            rope_deltas (LongTensor | None): RoPE position deltas.
            cache_position (LongTensor | None): Cache position indices.
            second_per_grid_ts (Tensor | None): Temporal information for video frames.
            **kwargs: Extra args forwarded to underlying model.

        Returns:
            CausalLMOutputWithPast | tuple: HF-standard structure (logits, past_key_values, hidden_states, etc.).
                May also include past_image_embeds and past_logit_mask for PaDT caching.

        Notes:
            - Autocast(bfloat16) used for efficiency.
            - padding_side already set to 'left' in tokenizer at init.
            - Hidden states required for auxiliary alignment or feature extraction modules.
            - PaDT-specific: Supports dynamic vocabulary expansion and visual prototype caching.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                past_image_embeds=past_image_embeds,
                past_logit_mask=past_logit_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        return outputs
    
    
    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_padtvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Construct and tokenize multimodal chat-style inputs for Qwen2.5-VL (batched).

        Parameters:
            images (List[List[PIL.Image.Image]]): Length B, each element is list of PIL images
            instructions (List[str]): Length B, textual prompts
            solutions (List[str], optional): For training labels
            **kwargs: Reserved for future extensions

        Returns:
            BatchFeature: HF-standard structure with input_ids, attention_mask, pixel_values, etc.
        """
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Prepare text prompts
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        # Process vision and text
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        # Sanity check to avoid opaque CUDA index asserts when merge-size/tokenization mismatches.
        if "input_ids" in batch_input and "image_grid_thw" in batch_input:
            image_token_id = (
                getattr(self.model.config, "image_token_id", None)
                or getattr(getattr(self.model.config, "text_config", None), "image_token_id", None)
            )
            if image_token_id is not None:
                n_image_tokens = int((batch_input["input_ids"] == image_token_id).sum().item())
                n_image_features = int(
                    (batch_input["image_grid_thw"].cumprod(-1)[:, -1] // (self.processor.spatial_merge_size ** 2)).sum().item()
                )
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        "Image token/feature mismatch before model forward: "
                        f"tokens={n_image_tokens}, features={n_image_features}, "
                        f"processor_merge={self.processor.spatial_merge_size}, "
                        f"grid_thw={batch_input['image_grid_thw'].tolist()}"
                    )
        

        # # Handle labels for training
        # if solutions is not None:
        #     action_token_min = _ACTION_TOKEN_MIN
        #     action_token_max = _ACTION_TOKEN_MAX
        #     labels = batch_input['input_ids'].clone()
            
        #     for i in range(labels.size(0)):
        #         seq = labels[i]
        #         mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        #         nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
        #         if nonzero_indices.numel() > 0:
        #             first_action_index = nonzero_indices[0].item()
        #             seq[:first_action_index] = IGNORE_INDEX
        #         else:
        #             seq[:] = IGNORE_INDEX
            
        #     labels[labels == self.processor.tokenizer.pad_token_id] = -100
        #     batch_input['labels'] = labels

        return batch_input.to(self.model.device)
    def build_padtvl_vlm_inputs(self, images, vlm_instructions, **kwargs):
        """
        Construct and tokenize multimodal chat-style inputs for Qwen2.5-VL (batched).

        Parameters:
            images (List[List[PIL.Image.Image]]): Length B, each element is list of PIL images
            vlm_instructions (List[str]): Length B, textual prompts for VLM
            solutions (List[str], optional): For training labels
            **kwargs: Reserved for future extensions

        Returns:
            BatchFeature: HF-standard structure with input_ids, attention_mask, pixel_values, etc.
        """
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(vlm_instructions), "Images and instructions must have the same length"
        
        for imgs, instruction in zip(images, vlm_instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            messages.append(msg)

        # Prepare text prompts
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]


        return texts


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
    
    model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    cfg.framework.qwenvl.base_vlm = model_id

    model = _PaDT_VL_Interface(config=cfg)
    pass

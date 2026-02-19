import os
import inspect
import warnings
import deepspeed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from dataclasses import dataclass
from transformers.cache_utils import Cache
from typing import Optional, Union, Callable, List, Tuple
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import is_deepspeed_zero3_enabled, is_fsdp_managed_module
from transformers.generation.utils import GenerateNonBeamOutput, GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VisionTransformerPretrainedModel


try:
    from transformers.generation.utils import is_torchdynamo_compiling
except ImportError:
    # Define a fallback function if it doesn't exist in your transformers version
    def is_torchdynamo_compiling():
        return False


@dataclass
class CustomGenerateEncoderDecoderOutput(GenerateEncoderDecoderOutput):
    past_image_embeds: Optional[torch.FloatTensor] = None,
    past_logit_mask: Optional[torch.BoolTensor] = None,
    # past_high_res_image_embeds: Optional[torch.FloatTensor] = None,
    # past_visual_pe: Optional[torch.FloatTensor] = None,

@dataclass
class CustomGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    past_image_embeds: Optional[torch.FloatTensor] = None
    past_logit_mask: Optional[torch.BoolTensor] = None
    # past_high_res_image_embeds: Optional[torch.FloatTensor] = None
    # past_visual_pe: Optional[torch.FloatTensor] = None


# def custom_visual_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
#     """
#     Args:
#         hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
#             The final hidden states of the model.
#         grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
#             The temporal, height and width of feature shape of each image in LLM.

#     Returns:
#         `torch.Tensor`: hidden_states.
#     """
#     hidden_states = self.patch_embed(hidden_states)
#     rotary_pos_emb = self.rot_pos_emb(grid_thw)
#     window_index, cu_window_seqlens = self.get_window_index(grid_thw)
#     cu_window_seqlens = torch.tensor(
#         cu_window_seqlens,
#         device=hidden_states.device,
#         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
#     )
#     cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

#     seq_len, _ = hidden_states.size()
#     hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
#     hidden_states = hidden_states[window_index, :, :]
#     hidden_states = hidden_states.reshape(seq_len, -1)
#     rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
#     rotary_pos_emb = rotary_pos_emb[window_index, :, :]
#     rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
#     emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#     position_embeddings = (emb.cos(), emb.sin())

#     cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
#         dim=0,
#         # Select dtype based on the following factors:
#         #  - FA2 requires that cu_seqlens_q must have dtype int32
#         #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
#         # See https://github.com/huggingface/transformers/pull/34852 for more information
#         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
#     )
#     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

#     for layer_num, blk in enumerate(self.blocks):
#         if layer_num in self.fullatt_block_indexes:
#             cu_seqlens_now = cu_seqlens
#         else:
#             cu_seqlens_now = cu_window_seqlens
#         if self.gradient_checkpointing and self.training:
#             hidden_states = self._gradient_checkpointing_func(
#                 blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
#             )
#         else:
#             hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
    
#     high_res_hidden_states = hidden_states
#     hidden_states = self.merger(hidden_states)
#     reverse_indices = torch.argsort(window_index)
#     hidden_states = hidden_states[reverse_indices, :]

#     return hidden_states, high_res_hidden_states, position_embeddings

# Qwen2_5_VisionTransformerPretrainedModel.forward = custom_visual_forward

class ZeroInitLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PaDTForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # self.use_visual_prototype_projection = self.config.use_visual_prototype_projection if hasattr(self.config, 'use_visual_prototype_projection') else True
        self.use_visual_prototype_projection = True

        if self.use_visual_prototype_projection:
            self.lora_r = 64
            # Robustly resolve hidden size across config variants
            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None and hasattr(self.config, "text_config"):
                hidden_size = getattr(self.config.text_config, "hidden_size", None)
            if hidden_size is None and hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
                hidden_size = self.model.embed_tokens.weight.shape[1]
            if hidden_size is None:
                raise AttributeError("Cannot determine hidden_size from config or model")

            self.vis_norm = ZeroInitLayerNorm(hidden_size)
            self.vis_proj = nn.Sequential(
                nn.Linear(hidden_size, self.lora_r, bias=False),
                nn.Linear(self.lora_r, hidden_size, bias=False),
            )
            
        self.rope_deltas = None

        # if '_attn_implementation' in config:
        #     self.config.vl_decoder['attn_implementation'] = config._attn_implementation
        # self.config.vl_decoder['spatial_merge_size'] = config.vision_config.spatial_merge_size
        # self.config.vl_decoder['llm_hidden_state'] = config.hidden_size

        # self.vl_decoder = PaDTDecoder(self.config.vl_decoder, self.dtype)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ported from transformers Qwen2_5_VLModel.get_rope_index.

        Computes 3D RoPE indices for vision tokens and 1D indices for text tokens,
        returning both the position_ids (shape: [3, B, T]) and rope deltas used
        for cache offsetting."""

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = getattr(self.config, "image_token_id", None) or getattr(getattr(self.config, "text_config", None), "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None) or getattr(getattr(self.config, "text_config", None), "video_token_id", None)
        vision_start_token_id = getattr(self.config, "vision_start_token_id", None) or getattr(getattr(self.config, "text_config", None), "vision_start_token_id", None)
        mrope_position_deltas: List[torch.Tensor] = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, ids in enumerate(total_input_ids):
                ids = ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
                vision_tokens = ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = ids.tolist()
                llm_pos_ids_list: List[torch.Tensor] = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        second_per_grid_t = (
                            second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )
                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    t_index = time_tensor.long().flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ZeroInitLayerNorm):
            with torch.no_grad():
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,

        past_image_embeds: Optional[torch.FloatTensor] = None,
        past_logit_mask: Optional[torch.BoolTensor] = None,

        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
# diff
        # ── 兼容 Qwen2.5-VL 嵌套 config：统一获取 vocab_size / image_token_id / video_token_id ──
        def _cfg(name, default=None):
            v = getattr(self.config, name, None)
            if v is None and hasattr(self.config, "text_config"):
                v = getattr(self.config.text_config, name, None)
            return v if v is not None else default

        vocab_size = _cfg("vocab_size")
        if vocab_size is None:
            vocab_size = self.get_input_embeddings().weight.shape[0]
        image_token_id = _cfg("image_token_id")
        video_token_id = _cfg("video_token_id")

        # ── 获取 visual encoder（Qwen2.5-VL 里在 self.model.visual）──
        visual_encoder = getattr(self, "visual", None)
        if visual_encoder is None and hasattr(self, "model") and hasattr(self.model, "visual"):
            visual_encoder = self.model.visual

        if inputs_embeds is None:
            if pixel_values is not None:
                if visual_encoder is None:
                    raise AttributeError("PaDTForConditionalGeneration: cannot find visual encoder (self.visual / self.model.visual)")

                if image_grid_thw is None:
                    raise ValueError("pixel_values is provided but image_grid_thw is None")
                if pixel_values.ndim >= 1 and image_grid_thw.ndim == 2 and image_grid_thw.shape[-1] == 3:
                    # Fail fast with a clear error when processor-side patch flattening and grid metadata diverge.
                    expected_tokens = int(image_grid_thw.cumprod(-1)[:, -1].sum().item())
                    got_tokens = int(pixel_values.shape[0])
                    if got_tokens != expected_tokens:
                        raise ValueError(
                            "Vision input mismatch before visual encoder: "
                            f"pixel_values_tokens={got_tokens}, expected_from_grid={expected_tokens}, "
                            f"grid_thw={image_grid_thw.tolist()}, "
                            f"spatial_merge_size={getattr(self.config.vision_config, 'spatial_merge_size', 'NA')}"
                        )

                visual_dtype = getattr(visual_encoder, "dtype", None)
                if visual_dtype is None:
                    visual_dtype = next(visual_encoder.parameters()).dtype

                pixel_values = pixel_values.type(visual_dtype)
                # 提取特征
                # Qwen2.5-VL visual 返回 BaseModelOutputWithPooling:
                #   last_hidden_state = merger 前 (vision_hidden_size=1280)
                #   pooler_output     = merger 后 (llm_hidden_size=2048)
                # 我们需要 pooler_output（已投射到 LLM 空间）
                vis_out = visual_encoder(pixel_values, grid_thw=image_grid_thw)
                if isinstance(vis_out, torch.Tensor):
                    image_embeds = vis_out
                elif hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None:
                    image_embeds = vis_out.pooler_output
                else:
                    image_embeds = vis_out.last_hidden_state

                # 视觉特征投影到视觉原型
                if self.use_visual_prototype_projection:
                    image_prototypes = self.vis_norm(image_embeds)
                    image_prototypes = image_prototypes + self.vis_proj(image_prototypes)
                else:
                    image_prototypes = image_embeds.clone()

                # 动态扩展词表
                embed_tokens = self.get_input_embeddings().weight
                extended_embed_tokens = torch.cat([embed_tokens, image_prototypes], dim=0)

                # 限制词表预测范围：按 batch 维度，而不是按图片数量。
                batch_size = input_ids.shape[0]
                logit_mask = extended_embed_tokens.new_zeros((batch_size, extended_embed_tokens.shape[0]), dtype=torch.bool)
                logit_mask[:, :vocab_size] = True

                # 按输入序列中图片 token 的顺序，为每个样本分配对应的图片原型区间
                image_token_counts = (input_ids == image_token_id).sum(dim=1)
                cursor = 0
                for b, count in enumerate(image_token_counts.tolist()):
                    if count > 0:
                        logit_mask[b, vocab_size + cursor : vocab_size + cursor + count] = True
                        cursor += count

                if cursor != image_embeds.shape[0]:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens per batch total {cursor}, features {image_embeds.shape[0]}"
                    )

                assert input_ids.max() < extended_embed_tokens.shape[0]
                # 输入的inputs_ids映射到新的词表
                inputs_embeds = extended_embed_tokens[input_ids]

                # 确保图片token和图片特征数量匹配
                n_image_tokens = (input_ids == image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                # 用图片特征替换图片token的embedding
                image_mask = (
                    (input_ids == image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # decode阶段复用过去的图片特征
            elif past_image_embeds is not None:
                image_prototypes = past_image_embeds.detach()
                #high_res_image_embeds, visual_pe = past_high_res_image_embeds.detach(), past_visual_pe
                logit_mask = past_logit_mask.detach()
                
                embed_tokens = self.get_input_embeddings().weight

                extended_embed_tokens = torch.cat([embed_tokens, image_prototypes], dim=0)
                inputs_embeds = extended_embed_tokens[input_ids]
            # 纯文本输入
            else:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                logit_mask = inputs_embeds.new_ones((input_ids.shape[0], vocab_size), dtype=torch.bool)
# 
            if pixel_values_videos is not None:
                if visual_encoder is None:
                    raise AttributeError("PaDTForConditionalGeneration: cannot find visual encoder for video")
                v_dtype = getattr(visual_encoder, "dtype", None) or next(visual_encoder.parameters()).dtype
                pixel_values_videos = pixel_values_videos.type(v_dtype)
                vid_out = visual_encoder(pixel_values_videos, grid_thw=video_grid_thw)
                if isinstance(vid_out, torch.Tensor):
                    video_embeds = vid_out
                elif hasattr(vid_out, "pooler_output") and vid_out.pooler_output is not None:
                    video_embeds = vid_out.pooler_output
                else:
                    video_embeds = vid_out.last_hidden_state
                n_video_tokens = (input_ids == video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts, attention_mask=attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # 扩展后的输入embedding
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]

        # diff

        if image_prototypes is not None:
            if self.config.tie_word_embeddings:
                logits = hidden_states @ extended_embed_tokens.T
            else:
                lm_weight = torch.cat([self.lm_head.weight, image_prototypes], dim=0)
                logits = hidden_states @ lm_weight.T
        # 
        else:
            logits = self.lm_head(hidden_states)  # Hidden_State: L * D, lm_head: V * D
        
        # diff

        logits.masked_fill_((~logit_mask[:, None, :]).expand(-1, logits.shape[1], -1), -float('inf'))
# 
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        ret = Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        # diff
        if past_image_embeds is not None:
            setattr(ret, "past_image_embeds", past_image_embeds)
            setattr(ret, "past_logit_mask", past_logit_mask)
            # setattr(ret, "past_high_res_image_embeds", high_res_image_embeds)
            # setattr(ret, "past_visual_pe", visual_pe)
        elif image_prototypes is not None:
            setattr(ret, "past_image_embeds", image_prototypes.clone())
            setattr(ret, "past_logit_mask", logit_mask.clone())
            # setattr(ret, "past_high_res_image_embeds", high_res_image_embeds.clone())
            # setattr(ret, "past_visual_pe", visual_pe)
        # 
        return ret
    
 
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        "https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py"

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        # self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # diff 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample_vision_token(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and not is_torchdynamo_compiling()
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _sample_vision_token(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        "https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py"
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            is_compileable = is_compileable and not self.generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            # , cur_len=cur_len, max_length=max_length
            this_peer_finished, synced_gpus, device=input_ids.device
        ): 
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # diff

            model_inputs.update({'past_image_embeds': model_kwargs['past_image_embeds']} if 'past_image_embeds' in model_kwargs else {})
            model_inputs.update({'past_logit_mask': model_kwargs['past_logit_mask']} if 'past_logit_mask' in model_kwargs else {})
            # model_inputs.update({'past_high_res_image_embeds': model_kwargs['past_high_res_image_embeds']} if 'past_high_res_image_embeds' in model_kwargs else {})
            # model_inputs.update({'past_visual_pe': model_kwargs['past_visual_pe']} if 'past_visual_pe' in model_kwargs else {})
            #

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
# diff
            if hasattr(outputs, 'past_image_embeds'):
                model_kwargs['past_image_embeds'] = outputs.past_image_embeds
            if hasattr(outputs, 'past_logit_mask'):
                model_kwargs['past_logit_mask'] = outputs.past_logit_mask
            # if hasattr(outputs, 'past_high_res_image_embeds'):
            #     model_kwargs['past_high_res_image_embeds'] = outputs.past_high_res_image_embeds
            # if hasattr(outputs, 'past_visual_pe'):
            #     model_kwargs['past_visual_pe'] = outputs.past_visual_pe
#
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()
        
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return CustomGenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    
                    past_image_embeds=model_kwargs.get("past_image_embeds"),
                    past_logit_mask=model_kwargs.get("past_logit_mask"),
                    # past_high_res_image_embeds=model_kwargs.get("past_high_res_image_embeds"),
                    # past_visual_pe=model_kwargs.get("past_visual_pe"),
                )
            else:
                return CustomGenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),

                    past_image_embeds=model_kwargs.get("past_image_embeds"),
                    past_logit_mask=model_kwargs.get("past_logit_mask"),
                    # past_high_res_image_embeds=model_kwargs.get("past_high_res_image_embeds"),
                    # past_visual_pe=model_kwargs.get("past_visual_pe"),
                )
        else:
            return input_ids

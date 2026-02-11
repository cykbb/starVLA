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
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################

@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
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
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config (handle Qwen2.5-VL config without top-level hidden_size)
        num_vl_layers = 36
        vl_config = self.qwen_vl_interface.model.config
        llm_hidden_size = getattr(vl_config, "hidden_size", None)
        if llm_hidden_size is None and hasattr(vl_config, "text_config"):
            llm_hidden_size = getattr(vl_config.text_config, "hidden_size", None)
        if llm_hidden_size is None:
            model_core = getattr(self.qwen_vl_interface.model, "model", None)
            embed_tokens = getattr(model_core, "embed_tokens", None)
            if embed_tokens is not None and hasattr(embed_tokens, "weight"):
                llm_hidden_size = embed_tokens.weight.shape[1]
        if llm_hidden_size is None:
            raise AttributeError("Cannot resolve llm hidden_size from Qwen2.5-VL config/model")
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]
        # agent_images = [[example["image"][0]] for example in examples]  #  [B, [Primary Camera only]] - 只使用第三视角
        instructions = [example.get("lang", example.get("language")) for example in examples]  # [B, str]
        if any(instr is None for instr in instructions):
            missing = [i for i, instr in enumerate(instructions) if instr is None]
            raise KeyError(f"Missing instruction key ('lang' or 'language') in examples at indices {missing}")
        # print(f"instruction: {instructions[0]}")
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 取与 DiT 层数匹配的最后 N 层隐藏态，按层喂给 DiT
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
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
            repeated_diffusion_steps = 2 # NO repeat for big action FM
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



        return {"action_loss": action_loss}



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
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
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
    

    model = Qwen_PI(cfg)
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
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
# """
# Qwen-GROOT Framework
# A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
# Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
# """
# from typing import List, Optional, Tuple
# from tqdm import tqdm
# import torch
# import numpy as np
# from PIL import Image

# from starVLA.training.trainer_utils import initialize_overwatch
# from deployment.model_server.tools.image_tools import to_pil_preserve

# logger = initialize_overwatch(__name__)

# IGNORE_INDEX = -100

# from starVLA.model.framework.base_framework import baseframework
# from starVLA.model.modules.vlm import get_vlm_model
# from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
# from starVLA.training.trainer_utils.trainer_tools import resize_images
# from starVLA.model.tools import FRAMEWORK_REGISTRY


# def _is_rank0() -> bool:
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         return torch.distributed.get_rank() == 0
#     return True


# def debug_print_qwen_inputs(qwen_inputs, processor=None, max_decode_tokens: int = 256, title: str = "qwen_inputs"):
#     """Print keys + tensor stats; decode input_ids[0] tail for special tokens."""
#     if not _is_rank0():
#         return

#     print(f"\n========== {title} DEBUG ==========")
#     if hasattr(qwen_inputs, "keys"):
#         keys = list(qwen_inputs.keys())
#     elif hasattr(qwen_inputs, "data") and hasattr(qwen_inputs.data, "keys"):
#         keys = list(qwen_inputs.data.keys())
#     else:
#         keys = []
#         print("WARNING: qwen_inputs is not dict-like, type =", type(qwen_inputs))

#     print("Keys:", keys)

#     for k in keys:
#         v = qwen_inputs[k]
#         if torch.is_tensor(v):
#             shape = tuple(v.shape)
#             dtype = v.dtype
#             device = v.device
#             msg = f"- {k:20s} dtype={str(dtype):12s} shape={shape} device={device}"
#             try:
#                 if v.numel() > 0 and v.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
#                     msg += f"  min={v.min().item():.4g} max={v.max().item():.4g}"
#                 elif v.numel() > 0 and v.dtype in (
#                     torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool
#                 ):
#                     uniq = torch.unique(v.detach().cpu())
#                     msg += f"  unique[:20]={uniq[:20].tolist()}"
#             except Exception:
#                 pass
#             print(msg)
#         else:
#             print(f"- {k:20s} type={type(v)}")

#     if processor is not None and "input_ids" in keys:
#         try:
#             ids = qwen_inputs["input_ids"]
#             if torch.is_tensor(ids) and ids.ndim == 2 and ids.shape[0] > 0:
#                 ids0 = ids[0].detach().cpu()
#                 if ids0.numel() > max_decode_tokens:
#                     ids0 = ids0[-max_decode_tokens:]
#                 text0 = processor.tokenizer.decode(ids0, skip_special_tokens=False)
#                 print("\n--- Decoded input_ids[0] (tail) ---")
#                 print(text0)
#                 print("--- end decode ---")
#         except Exception as e:
#             print("WARNING: decode failed:", repr(e))

#     print(f"========== END {title} DEBUG ==========\n")


# def debug_alignment_checks(qwen_inputs, model, title: str = "alignment"):
#     """Extra checks: sum(T*H*W) == pixel_values rows; count image_token_id per sample."""
#     if not _is_rank0():
#         return

#     print(f"\n========== {title} CHECKS ==========")

#     # 1) sum(T*H*W) vs pixel_values rows
#     if "image_grid_thw" in qwen_inputs and "pixel_values" in qwen_inputs:
#         grid = qwen_inputs["image_grid_thw"]
#         pv = qwen_inputs["pixel_values"]
#         if torch.is_tensor(grid) and torch.is_tensor(pv) and grid.ndim == 2 and grid.shape[1] == 3:
#             # grid: (num_images, 3) -> sum over images of T*H*W
#             patch_count = (grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item()
#             print(f"sum(T*H*W) = {patch_count}")
#             print(f"pixel_values rows = {pv.shape[0]}")
#             print(f"match? {patch_count == pv.shape[0]}")

#     # 2) count image_token_id (<|image_pad|>) per sample in input_ids
#     if "input_ids" in qwen_inputs:
#         ids = qwen_inputs["input_ids"]
#         if torch.is_tensor(ids) and ids.ndim == 2:
#             image_token_id = getattr(model.config, "image_token_id", None)
#             video_token_id = getattr(model.config, "video_token_id", None)
#             print(f"model.config.image_token_id = {image_token_id}")
#             print(f"model.config.video_token_id = {video_token_id}")

#             if image_token_id is not None:
#                 n_img_tokens = (ids == image_token_id).sum(dim=1)
#                 print("image_pad tokens per sample (first 16):", n_img_tokens[:16].tolist())

#             if video_token_id is not None:
#                 n_vid_tokens = (ids == video_token_id).sum(dim=1)
#                 print("video_pad tokens per sample (first 16):", n_vid_tokens[:16].tolist())

#     print(f"========== END {title} CHECKS ==========\n")


# @FRAMEWORK_REGISTRY.register("QwenPI")
# class Qwen_PI(baseframework):
#     def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
#         super().__init__()
#         self.config = config
#         self.qwen_vl_interface = get_vlm_model(config=self.config)

#         # dynamic get llm config
#         num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size
#         self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
#         self.config.framework.qwenvl.num_vl_layers = num_vl_layers

#         self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

#         self.future_action_window_size = config.framework.action_model.future_action_window_size
#         self.past_action_window_size = config.framework.action_model.past_action_window_size
#         self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

#         # Debug toggle
#         self.DEBUG_QWEN = True

#     def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
#         batch_images = [example["image"] for example in examples]
#         instructions = [example["lang"] for example in examples]
#         actions = [example["action"] for example in examples]
#         state = [example["state"] for example in examples] if "state" in examples[0] else None

#         # Step 1: Build Qwen inputs
#         qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

#         # Debug: print + alignment checks
#         if self.DEBUG_QWEN:
#             proc = getattr(self.qwen_vl_interface, "processor", None)
#             debug_print_qwen_inputs(qwen_inputs, processor=proc, title="qwen_inputs (after build)")
#             debug_alignment_checks(qwen_inputs, model=self.qwen_vl_interface.model, title="qwen_inputs alignment")

#         # Step 2: Qwen forward
#         with torch.autocast("cuda", dtype=torch.bfloat16):
#             qwenvl_outputs = self.qwen_vl_interface(
#                 **qwen_inputs,
#                 output_attentions=False,
#                 output_hidden_states=True,
#                 return_dict=True,
#             )
#             all_hidden = qwenvl_outputs.hidden_states
#             expected_layers = len(self.action_model.model.transformer_blocks)
#             vl_embs_list = list(all_hidden[-expected_layers:])
#             base_hidden = vl_embs_list[-1]

#         # Step 3: action head loss
#         with torch.autocast("cuda", dtype=torch.float32):
#             actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
#             actions_target = actions[:, -(self.future_action_window_size + 1):, :]

#             repeated_diffusion_steps = (
#                 self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
#             )
#             repeated_diffusion_steps = 2

#             actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
#             vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

#             state_repeated = None
#             if state is not None:
#                 state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
#                 state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

#             action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)

#         return {"action_loss": action_loss}

#     @torch.inference_mode()
#     def predict_action(self, examples: List[dict] = None, **kwargs: str) -> np.ndarray:
#         batch_images = [to_pil_preserve(example["image"]) for example in examples]
#         instructions = [example["lang"] for example in examples]
#         state = [example["state"] for example in examples] if "state" in examples[0] else None

#         train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
#         if train_obs_image_size:
#             batch_images = resize_images(batch_images, target_size=train_obs_image_size)

#         qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

#         if self.DEBUG_QWEN:
#             proc = getattr(self.qwen_vl_interface, "processor", None)
#             debug_print_qwen_inputs(qwen_inputs, processor=proc, title="qwen_inputs (predict_action)")
#             debug_alignment_checks(qwen_inputs, model=self.qwen_vl_interface.model, title="qwen_inputs alignment (predict_action)")

#         with torch.autocast("cuda", dtype=torch.bfloat16):
#             qwenvl_outputs = self.qwen_vl_interface(
#                 **qwen_inputs,
#                 output_attentions=False,
#                 output_hidden_states=True,
#                 return_dict=True,
#             )
#             all_hidden = qwenvl_outputs.hidden_states
#             expected_layers = len(self.action_model.model.transformer_blocks)
#             vl_embs_list = list(all_hidden[-expected_layers:])
#             base_hidden = vl_embs_list[-1]

#         state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
#         with torch.autocast("cuda", dtype=torch.float32):
#             pred_actions = self.action_model.predict_action(vl_embs_list, state)

#         normalized_actions = pred_actions.detach().cpu().numpy()
#         return {"normalized_actions": normalized_actions}


# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     import debugpy
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml")
#     args, clipargs = parser.parse_known_args()

#     debugpy.listen(("0.0.0.0", 10092))
#     print("🔍 Rank 0 waiting for debugger attach on port 10092...")
#     debugpy.wait_for_client()

#     cfg = OmegaConf.load(args.config_yaml)
#     cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

#     model = Qwen_PI(cfg)
#     print(model)

#     image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
#     sample = {
#         "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
#         "image": [image, image],  # two views
#         "lang": "This is a fake instruction for testing.",
#         "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
#     }

#     batch = [sample, sample] * 8  # make B=16 to match your log
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     forward_output = model(batch)
#     action_loss = forward_output["action_loss"]
#     print(f"Action Loss: {action_loss.item()}")

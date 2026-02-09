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

        # dynamic get llm config
        num_vl_layers, llm_hidden_size = 36, self.padt_vl_interface.model.config.hidden_size
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
        
        # 检查是否有 answers 数据
        if "answers" not in examples[0]:
            return None
            
        # 构建 prompt 和 completion，并准备 solutions（用于 bbox 等信息）
        prompts = []
        completions = []
        solutions = []
        
        for example in examples:
            answers_data = example["answers"]
            
            # 从 answers 中提取对话
            if "conversations" in answers_data and len(answers_data["conversations"]) > 0:
                # 使用对话作为 prompt
                conversation = answers_data["conversations"][0]
                prompt = [{"role": "user", "content": conversation["value"]}]
                prompts.append(prompt)
                
                # 使用 answer_template 作为 completion
                completion = answers_data.get("answer_template", "")
                
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
                    obj_indices = re.findall(pattern, completion)
                    
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
                            
                            # 如果找到了该物体的信息（可能来自1-2个视角）
                            if obj_info_list:
                                objects_info[obj_idx] = obj_info_list
                
                # 构建 solution 数据结构
                solution = {
                    'text': completion,
                    'objects': objects_info
                }
                solutions.append(solution)
                completions.append(completion)
            else:
                # 如果没有对话数据，跳过
                return None
        
        # 使用 padt_vl_interface 的 tokenizer 处理
        prompt_texts = [self.padt_vl_interface.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        ) for prompt in prompts]
        
        # 使用所有视角的图片（和padt.py的forward逻辑一致）
        # batch_images: List[List[PIL.Image]], 展平为一维列表供processor处理
        from qwen_vl_utils import process_vision_info
        
        # 构建messages用于process_vision_info
        messages_for_vision = []
        for imgs in batch_images:
            content = [{"type": "image", "image": img} for img in imgs]
            messages_for_vision.append([{"role": "user", "content": content}])
        
        image_inputs, _ = process_vision_info(messages_for_vision)
        
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
        
        # Tokenize completions
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
        
        # Prepare multimodal inputs
        multimodal_inputs = {
            'image_grid_thw': prompt_inputs.get('image_grid_thw'),
            'pixel_values': prompt_inputs.get('pixel_values')
        }
        
        # Forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_output = self.padt_vl_interface.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **multimodal_inputs
            )
        
        # Compute token loss
        logits = model_output.logits[:, prompt_length-1:-1, :]  # (B, L, V)
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
        instructions = [example["lang"] for example in examples]  # [B, str]
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

        # 构建返回字典
        output_dict = {"action_loss": action_loss}
        
        # 计算 VLM loss（如果需要且数据可用）
        if compute_vlm_loss:
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
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
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

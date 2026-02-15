import torch
import torch.distributed as dist
import logging
from transformers.tokenization_utils import AddedToken

logger = logging.getLogger(__name__)

class VisonTextProcessingClass(object):
    def __init__(self, processing_class, spatial_merge_size=1):
        self.processing_class = processing_class
        self.spatial_merge_size = spatial_merge_size
        self.model_embed_token_size = len(processing_class.tokenizer.get_vocab())

    def __getattr__(self, name: str):
        if hasattr(self.processing_class, name):
            return getattr(self.processing_class, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def prepare(self, model_embed_token_size):
        self.model_embed_token_size = model_embed_token_size
        need_pad_size = model_embed_token_size - len(self.tokenizer.get_vocab())
        assert '<|empty_token_0|>' in self.tokenizer.vocab or need_pad_size > 0
        if need_pad_size > 0:
            self.tokenizer.add_tokens([AddedToken("<|empty_token_%d|>" % i, lstrip=False, rstrip=False, special=True, normalized=False) for i in range(need_pad_size)])
        return True
    
    def set_image_grid_thw(self, image_grid_thw):
        # add visual patch tokenization into processing_class
        max_visual_patch_num = image_grid_thw.cumprod(-1).max(dim=0)[0][-1] // (self.spatial_merge_size) ** 2
        if len(self.processing_class.tokenizer.get_vocab()) - self.model_embed_token_size < max_visual_patch_num:
            self.processing_class.tokenizer.add_tokens([AddedToken("<|VRT_%d|>" % i, lstrip=False, rstrip=False, special=False, normalized=False) for i in range(len(self.processing_class.tokenizer.get_vocab()) - self.model_embed_token_size, max_visual_patch_num)])
        return True

    def __call__(self, *args, **kwargs):
        parent_ret = self.processing_class(*args, **kwargs)
        if 'image_grid_thw' in parent_ret:
            self.set_image_grid_thw(parent_ret['image_grid_thw'])
        return parent_ret
    # 按“一个 sample = 两张图”先算每个 sample 的总 patch 数，再把 batch 里后续样本整体向后偏移：sample0: 0..511，sample1: 512..1023，...
    def assign_to_global_vrt_id(self, input_ids, image_grid_thw, images_per_sample=None):
        debug_vrt = bool(getattr(self, "debug_vrt", False))
        debug_max_calls = int(getattr(self, "debug_vrt_max_calls", 20))
        debug_preview = int(getattr(self, "debug_vrt_token_preview", 12))
        debug_call_count = int(getattr(self, "_debug_vrt_assign_call_count", 0))
        is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        debug_this_call = debug_vrt and is_main and debug_call_count < debug_max_calls
        if debug_this_call:
            self._debug_vrt_assign_call_count = debug_call_count + 1

        visual_patch_mask = input_ids >= self.model_embed_token_size
        before_vrt = input_ids[visual_patch_mask].clone() if debug_this_call and visual_patch_mask.any() else None
        if visual_patch_mask.sum() > 0:
            per_image_patches = image_grid_thw.cumprod(-1)[:, -1] // (self.spatial_merge_size ** 2)

            if images_per_sample is not None and len(image_grid_thw) != input_ids.shape[0]:
                sample_total_patches = []
                idx = 0
                for n_imgs in images_per_sample:
                    sample_total_patches.append(per_image_patches[idx:idx + n_imgs].sum())
                    idx += n_imgs
                sample_total_patches = torch.stack(sample_total_patches)
                each_sample_image_patches = torch.nn.functional.pad(
                    sample_total_patches.cumsum(dim=-1), (1, 0), "constant", 0
                )
            else:
                each_sample_image_patches = torch.nn.functional.pad(
                    per_image_patches.cumsum(dim=-1), (1, 0), "constant", 0
                )

            each_sample_image_patches_ = each_sample_image_patches[:-1, None].expand(-1, input_ids.shape[1])
            input_ids[visual_patch_mask] += each_sample_image_patches_[visual_patch_mask]
            if debug_this_call:
                after_vrt = input_ids[visual_patch_mask]
                logger.info(
                    "[VRTDBG] assign_to_global_vrt_id call=%s per_image_patches=%s sample_offsets=%s "
                    "before_vrt_head=%s after_vrt_head=%s vrt_count=%s",
                    self._debug_vrt_assign_call_count,
                    per_image_patches.tolist(),
                    each_sample_image_patches[:-1].tolist(),
                    before_vrt[:debug_preview].tolist() if before_vrt is not None else [],
                    after_vrt[:debug_preview].tolist(),
                    int(after_vrt.numel()),
                )
        return input_ids
    
    def assign_to_local_vrt_id(self, input_ids, image_grid_thw, images_per_sample=None):
        visual_patch_mask = input_ids >= self.model_embed_token_size
        if visual_patch_mask.sum() > 0:
            per_image_patches = image_grid_thw.cumprod(-1)[:, -1] // (self.spatial_merge_size ** 2)
            
            if images_per_sample is not None and len(image_grid_thw) != input_ids.shape[0]:
                sample_total_patches = []
                idx = 0
                for n_imgs in images_per_sample:
                    sample_total_patches.append(per_image_patches[idx:idx + n_imgs].sum())
                    idx += n_imgs
                sample_total_patches = torch.stack(sample_total_patches)
                each_sample_image_patches = torch.nn.functional.pad(sample_total_patches.cumsum(dim=-1), (1, 0), 'constant', 0)
            else:
                each_sample_image_patches = torch.nn.functional.pad(per_image_patches.cumsum(dim=-1), (1, 0), 'constant', 0)
            
            each_sample_image_patches_ = each_sample_image_patches[:-1, None].expand(-1, input_ids.shape[1])
            input_ids[visual_patch_mask] -= each_sample_image_patches_[visual_patch_mask]
        return input_ids
    # 单个 sample、两张图、每图 16x16=256 patch 时：图0 用 0..255，图1 用 256..511
    def pid2vrt(self, patch_ids):
        if type(patch_ids) == int:
            patch_ids = [patch_ids]
        else:
            patch_ids = [int(i) for i in patch_ids]
        return ''.join(['<|VRT_%d|>' % i for i in patch_ids])


def parseVRTintoCompletion(processor, completion_ids, hidden_states, need_thinking_mask=None, image_prototype=None, image_grid_thw=None):
    # hidden_states: [N, Layers, BS, 1, D]
    dim = hidden_states[0][-1].shape[-1]
    ret_list = []
    ret_completions = []
    ret_labels = []
    ret_vrts = []
    ret_vrts_feats = []

    if image_grid_thw is not None:
        vision_patch_nums = torch.nn.functional.pad((image_grid_thw.cumprod(-1)[:, -1] // (processor.spatial_merge_size ** 2)).cumsum(-1), (1, 0), 'constant', 0)
    
    if need_thinking_mask is None:
        need_thinking_mask = torch.ones(len(completion_ids)).to(torch.bool)

    for batch_idx, completion in enumerate(completion_ids):
        completion_per_vob = processor.batch_decode(completion)
        completion_str = ''.join(completion_per_vob)
        ret_completions.append(completion_str)
        
        sample_ret_list = []
        sample_ret_labels = []
        sample_ret_vrts = []
        sample_ret_vrts_feats = []

        vob_idx = 0
        without_thinking = not need_thinking_mask[batch_idx].item()
        within_answer_tag = False
        within_object_name_tag = False
        object_idx = -1
        current_label = ""

        try:
            while vob_idx < len(completion_per_vob):
                if processor.tokenizer.eos_token in completion_per_vob[vob_idx]:
                    break
                if within_answer_tag is False and '<' in completion_per_vob[vob_idx] and '</' not in completion_per_vob[vob_idx] and 'answer' in completion_per_vob[vob_idx + 1] and '>' in completion_per_vob[vob_idx + 2]:
                    within_answer_tag = True
                    vob_idx += 3
                    continue
                if within_answer_tag is True or without_thinking:
                    if '</' in completion_per_vob[vob_idx] and 'answer' in completion_per_vob[vob_idx + 1]  and '>' in completion_per_vob[vob_idx + 2]:
                        within_answer_tag = False
                        break
                    else:
                        if '"' in completion_per_vob[vob_idx] and within_object_name_tag is False:
                            within_object_name_tag = True
                            current_label = completion_per_vob[vob_idx].split('"')[1]
                            vob_idx += 1
                            continue

                        if '"' in completion_per_vob[vob_idx] and within_object_name_tag is True:
                            within_object_name_tag = False
                            current_label += completion_per_vob[vob_idx].split('"')[0]
                            current_label = current_label.strip()
                            vob_idx += 1
                            continue
                        
                        if '<|VRT_' in completion_per_vob[vob_idx]:
                            within_object_name_tag = False

                            vrt_hidden_states = []
                            vrts_str = ""

                            while '<|VRT_' in completion_per_vob[vob_idx]:
                                vrt_hidden_states.append(hidden_states[vob_idx][-1][batch_idx])
                                vrts_str += completion_per_vob[vob_idx]
                                vob_idx += 1
                            
                            sample_ret_list.append(torch.cat(vrt_hidden_states, dim=0))
                            sample_ret_labels.append(current_label)
                            sample_ret_vrts.append(vrts_str)

                            if image_prototype is not None and image_grid_thw is not None:
                                vrts_ids = processor(text=vrts_str, return_tensors='pt')['input_ids'].to(image_grid_thw.device)[0, ...] + vision_patch_nums[batch_idx] - processor.model_embed_token_size
                                vrts_feats = image_prototype[vrts_ids]
                                sample_ret_vrts_feats.append(vrts_feats)
                            continue
                        
                        if within_object_name_tag:
                            current_label += completion_per_vob[vob_idx]
                vob_idx += 1
            ret_list.append(sample_ret_list)
            ret_labels.append(sample_ret_labels)
            ret_vrts.append(sample_ret_vrts)
            ret_vrts_feats.append(sample_ret_vrts_feats)
        except:
            ret_list.append([])
            ret_labels.append([])
            ret_vrts.append([])
            ret_vrts_feats.append([])
    return ret_completions, ret_list, ret_labels, ret_vrts, ret_vrts_feats

"""
处理image
"""
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
# ======================================================================================================
# 读取模型权重，推荐情况下用下面那个被注释掉的引入flashattn的做法
# ======================================================================================================
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# ======================================================================================================
# 读取用于处理数据中图像和文本的processor，细节如下：
#（1） 默认情况下，在qwen2.5 LM Decoder的输入中，一张图片最少占据4个token，最多占据16384个token
# (2) 你也可以自己权衡模型效果和计算成本，自行设定一张图片最少/最多占据的token数量，
#     然后把这个自定义值传入process初始化的参数重，例如：
#     min_pixels = 256*28*28，你希望一张图片最少占据256个token，由于每个token对应一块28*28的区域，所以这张图片至少拥有256*28*28个pixel
#     max_pixels = 1280*28*28，道理同上
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
# 
#     这里，取28是因为，最初的patch_size我们打算设为14，由此得到原始patch，这也是vit部分的输入。
#     但在vit输出层，为了进一步节省token，我们决定将 2*2 个patch合并起来作为一个token
#     这个token才是最后作为qwen2.5 LM Decoder的vision部分输入，所以是14*2 = 28
#     （更多细节参见后文对vit部分的解读）
# ======================================================================================================
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# ======================================================================================================
# 传入prompt
# （1）一个text可以对应多个image（把每个image表示成1个dict就好）
# （2）你还可以定制化地去设定针对这张图片的参数，例如你只想对这张图片改变min_pixels和max_pixels，你就可以
#     在这张图片对应的字典中去添加这两个参数，详情参见文档 https://github.com/QwenLM/Qwen2.5-VL
# ======================================================================================================
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# ====================================================================================================
# Preparation for inference
# 对 messages 做一些处理，主要是加上一些诸如特殊字符
#
# text返回结果：
#（1）增加了默认的sys_msg
#（2）每一个角色（system或者user）说的话，其开头和结尾分别添加<|im_start|>和<|im_end|>标记
#（3）image的表达方式为<|vision_start|><|image_pad|><|vision_end|，其中<|image_pad|>是预留给图像的位置
#
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# <|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>
# <|im_start|>assistant
# ====================================================================================================
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"text: {text}")

# ====================================================================================================
# 预处理图像和视频数据。这里以图像数据举例，视频数据处理见后文
# image_inputs: List[PIL.Image.Image], 列表长度为这个batch中对应的所有图片数量，这点非常重要
# 假设列表长度为1，那么image_inputs形如：[<PIL.Image.Image image mode=RGB size=2044x1372>]
# process_vision_info的代码请看：https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L352
#
# process_vision_info对每个image都做了如下处理：
#（1）检查每张图片的 max(h,w)/min(h,w)是否在阈值范围内，如果超过阈值则认为该图片高宽比太离谱，会直接抛出异常（当前阈值200）
#（2）通过四舍五入的方式，重新设置图片的 h 和 w 值，确保它们可以被28整除
#（3）如果这张图片太大，超过了上述 max_pixels 的范围，那么就在尽量维持其宽高比例不变的情况下，缩小其宽高
#（4）这张图片太小时用同样的方式放大其宽高
#（5）经过前面的4步，我们得到了这张图片最终理想的h和w值，我们采用resize的方式把图片按这个值缩放，就得到image_inputs中的每一个图片
# 你可以发现，这里没有经过任何的“多裁少pad操作”，你只是在缩放图片。
# 
# process_vision_info对每个video都做了如下处理：TODO
# ====================================================================================================
image_inputs, video_inputs = process_vision_info(messages)
print(f"image_inputs: {image_inputs}")
print(f"video_inputs: {video_inputs}")
# ====================================================================================================
# 假设这里我们做的是batch inference，有2个text，text0对应2张图， text1对应1张图。
# 那么最终inputs的形式如：       
# {
# input_ids尺寸是：(text_num, token_num), 这里我们已经算好了每张图片会占据多少个token，并用相应个数的<|image_pad|>在text文本里做了替换
# 'input_ids': tensor([[151644,   8948,    198,  ..., 151644,  77091,    198],
#                      [151644,   8948,    198,  ..., 151643, 151643, 151643]]), 
# 
# attention_mask尺寸是：(text_num, token_num)
# 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
#                           [1, 1, 1,  ..., 0, 0, 0]]), 0表示第2条数据做了padding
# 
#  pixel_values尺寸为：(image_num * grid_t * grid_h * grid_w, 
#                      channel * temporal_patch_size(2) * patch_size(14) * patch_size(14))，image_num是这个batch中的image数量
#  'pixel_values': tensor([[ 0.8501,  0.8501,  0.8647,  ...,  1.3922,  1.3922,  1.3922],
#                          [ 0.9376,  0.9376,  0.9376,  ...,  1.4491,  1.4491,  1.4491],
#                          [ 0.9084,  0.9376,  0.9376,  ...,  1.4065,  1.4207,  1.4207],
#                          ...,
#                          [-0.1280, -0.1280, -0.1426,  ..., -0.2431, -0.2715, -0.3000],
#                          [-0.3324, -0.3324, -0.3032,  ..., -0.3000, -0.2715, -0.2857],
#                          [-0.3762, -0.4054, -0.4054,  ..., -0.4279, -0.4422, -0.4564]]),
#         
# image_grid_thw尺寸是：(image_num, 3)，其中3分别表示这张图片的grid_t, grid_h, grid_w
# 'image_grid_thw': tensor([[  1,  98, 146],
#                           [  1,  98, 146],
#                           [  1,  98, 146]])
# }
# 到这一步为止，我们还没有对图像做具体的转token处理，这个应该是在model.forward中做的，我们只是对图像做了一些初步的resize，rescale等处理
# ====================================================================================================
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print(f"inputs_ids:{inputs['input_ids']}")
print(f"inputs.keys(): {inputs.keys()}")
inputs = inputs.to(model.device)
print(f"pixel_values.shape: {inputs['pixel_values'].shape}")
print(f"image_grid_thw.shape: {inputs['image_grid_thw'].shape}")
print(f"input_ids.shape: {inputs['input_ids'].shape}")
print(f"attention_mask.shape: {inputs['attention_mask'].shape}")

# ====================================================================================================
# Inference: Generation of the output
# ====================================================================================================
generated_ids = model.generate(**inputs, max_new_tokens=128)
print(f"generated_ids:{generated_ids}")
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print(f"generated_ids_trimmed:{generated_ids_trimmed}")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
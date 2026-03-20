import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ====================== 路径配置 ======================
MODEL_PATH = r"H:\新建文件夹"  # 你的剪枝模型路径
IMAGE_PATH = r"test.jpg"                           # 你的图片路径
# ====================================================

print("Loading model...")
# 尝试加载模型，确保视觉组件被正确加载
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 使用原始参数名
    device_map="auto",            # 自动分配设备
    low_cpu_mem_usage=True,
    trust_remote_code=True,       # 允许远程代码（有些模型需要）
)

# 加载处理器
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# ---------------------- 关键修复：检查processor是否支持图像 ----------------------
print(f"Processor type: {type(processor)}")
print(f"Has image processor: {hasattr(processor, 'image_processor')}")

# ---------------------- 方法1：使用chat template（推荐） ----------------------
# 构建对话格式的消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": "请详细描述这张图片，生成可用于AI绘画的高质量中文提示词，只输出结果，不要解释"}
        ]
    }
]

# 加载图片
image = Image.open(IMAGE_PATH).convert("RGB")

# 使用apply_chat_template处理
try:
    # 尝试使用新的对话格式
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print("Using chat template")
    print(f"Processed text: {text[:200]}...")  # 打印前200字符
    
    # 处理输入
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    )
    
except Exception as e:
    print(f"Chat template failed: {e}")
    print("Falling back to manual formatting...")
    
    # 方法2：手动添加图像占位符
    # Qwen3-VL 使用 <|vision_start|> 和 <|vision_end|> 包裹图像
    prompt = "<|vision_start|><|image_pad|><|vision_end|>请详细描述这张图片，生成可用于AI绘画的高质量中文提示词，只输出结果，不要解释"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

# 确保输入在正确的设备上
inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

# 打印输入信息以调试
print(f"\nInput keys: {inputs.keys()}")
if 'input_ids' in inputs:
    print(f"Input shape: {inputs['input_ids'].shape}")
if 'pixel_values' in inputs:
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")

# 生成参数
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.15,
    "pad_token_id": processor.tokenizer.eos_token_id,
}

# 推理
print("\nGenerating...")
with torch.no_grad():
    outputs = model.generate(**inputs, **generate_kwargs)

# 输出结果
result = processor.decode(outputs[0], skip_special_tokens=True)

# 清理输出，只保留模型生成的回答
if "<|im_start|>assistant" in result:
    result = result.split("<|im_start|>assistant")[-1].strip()
if result.startswith("请详细描述"):
    # 如果输出包含了原始输入，尝试只取新生成的部分
    result = result.split("请详细描述")[0].strip()

print("\n" * 2)
print("=" * 80)
print("📷 图片反推提示词结果")
print("=" * 80)
print(result)
print("=" * 80)

# 保存结果
with open("prompt_result.txt", "w", encoding="utf-8") as f:
    f.write(result)

# 可选：打印生成的token数量
print(f"\n✅ 生成完成！结果已保存到 prompt_result.txt")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from PIL import Image
import os
import time
import argparse
import json
from pathlib import Path

# ====================== 强制GPU配置 ======================
# 设置CUDA设备
torch.cuda.set_device(0)
device = torch.device("cuda:0")

print(f"🎮 使用GPU: {torch.cuda.get_device_name(0)}")
print(f"📊 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"🔢 PyTorch版本: {torch.__version__}")
print(f"🔢 CUDA版本: {torch.version.cuda}")

# 验证兼容层是否生效
arch_list = torch.cuda.get_arch_list()
print(f"📋 支持的CUDA架构: {arch_list}")

# ====================== 模型配置 ======================
MODEL_PATH = r"./Qwen3-VL-8B-pruned-8percent-bf16"
DTYPE = torch.bfloat16
# ======================================================

def clear_gpu_memory():
    """清理GPU显存"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"🧹 GPU显存 - 已分配: {allocated:.2f}GB, 缓存: {cached:.2f}GB")

def get_pruned_dimensions():
    """从剪枝日志获取实际的模型维度"""
    pruning_log_path = os.path.join(MODEL_PATH, 'pruning_log.json')
    if os.path.exists(pruning_log_path):
        with open(pruning_log_path, 'r') as f:
            pruning_log = json.load(f)
        
        first_layer = pruning_log.get('layer_0', {})
        if first_layer and 'gate_proj' in first_layer:
            new_dim = first_layer['gate_proj']['new'][0]
            print(f"从剪枝日志中获取到新的MLP维度: {new_dim}")
            return new_dim
    
    return 11304  # 默认值

def load_model_gpu():
    """加载模型到GPU"""
    print("\n🚀 正在加载模型到GPU...")
    start_time = time.time()
    
    clear_gpu_memory()
    
    # 获取剪枝后的维度
    pruned_dim = get_pruned_dimensions()
    
    # 加载并修改配置
    print("加载并修改模型配置...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # 修改中间层大小
    if hasattr(config, 'text_config'):
        if hasattr(config.text_config, 'intermediate_size'):
            print(f"原始 intermediate_size: {config.text_config.intermediate_size}")
            config.text_config.intermediate_size = pruned_dim
            print(f"修改后 intermediate_size: {config.text_config.intermediate_size}")
    elif hasattr(config, 'intermediate_size'):
        print(f"原始 intermediate_size: {config.intermediate_size}")
        config.intermediate_size = pruned_dim
        print(f"修改后 intermediate_size: {config.intermediate_size}")
    
    print("加载模型权重...")
    
    # 加载模型
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=DTYPE,
        device_map=None,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    
    # 移动到GPU
    model = model.to(device)
    model.eval()
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    load_time = time.time() - start_time
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载完成！耗时: {load_time:.2f}秒")
    print(f"📊 模型参数量: {total_params/1e9:.2f}B")
    print(f"📍 模型设备: {next(model.parameters()).device}")
    
    # 验证维度
    verify_model_structure(model)
    
    clear_gpu_memory()
    
    return model, processor

def verify_model_structure(model):
    """验证模型实际加载的维度"""
    print("\n🔍 验证模型实际结构:")
    
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            layers = model.model.language_model.layers
            first_layer = layers[0]
            
            if hasattr(first_layer, 'mlp'):
                mlp = first_layer.mlp
                
                if hasattr(mlp, 'gate_proj'):
                    gate_dim = mlp.gate_proj.weight.shape[0]
                    print(f"  gate_proj 实际维度: {gate_dim}")
                
                if hasattr(mlp, 'up_proj'):
                    up_dim = mlp.up_proj.weight.shape[0]
                    print(f"  up_proj 实际维度: {up_dim}")
                
                if hasattr(mlp, 'down_proj'):
                    down_dim = mlp.down_proj.weight.shape[1]
                    print(f"  down_proj 实际维度: {down_dim}")
    except Exception as e:
        print(f"验证过程中出错: {e}")
    
    print("✅ 模型结构验证完成")

def load_image(image_path):
    """加载图片"""
    try:
        if image_path.startswith(('http://', 'https://')):
            import requests
            from io import BytesIO
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise Exception(f"图片加载失败 {image_path}: {e}")

def generate_caption(model, processor, image, prompt_template=None, max_new_tokens=256):
    """生成图像描述"""
    
    if prompt_template is None:
        prompt_template = "Describe this image in detail, including objects, scenes, colors, and any text visible."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_template}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    before_mem = torch.cuda.memory_allocated() / 1024**3
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=DTYPE):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
    
    after_mem = torch.cuda.memory_allocated() / 1024**3
    
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return caption, after_mem - before_mem

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen3-VL 剪枝模型推理")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--prompt", type=str, default="detailed", 
                       choices=["simple", "detailed", "style", "objects", "scene", "action", "technical", "stable_diffusion"],
                       help="提示词类型")
    
    args = parser.parse_args()
    
    prompt_templates = {
        "simple": "What is in this image?",
        "detailed": "Describe this image in detail, including the main subject, background, colors, lighting, and composition.",
        "style": "Describe the artistic style, color palette, and visual techniques used in this image.",
        "objects": "List all objects, people, and items visible in this image.",
        "scene": "Describe the scene, setting, environment, and atmosphere in this image.",
        "action": "Describe the actions, interactions, and expressions in this image.",
        "technical": "Provide a technical description including composition, lighting, depth of field, and camera angle.",
        "stable_diffusion": "Write a detailed prompt suitable for Stable Diffusion that would generate this image."
    }
    
    try:
        # 加载模型
        model, processor = load_model_gpu()
        
        # 加载图片
        print(f"\n📸 加载图片: {args.image}")
        image = load_image(args.image)
        print(f"📏 图片尺寸: {image.size}")
        
        # 生成描述
        prompt = prompt_templates[args.prompt]
        print(f"📝 使用提示词类型: {args.prompt}")
        
        caption, mem_used = generate_caption(model, processor, image, prompt)
        
        print("\n" + "="*50)
        print("📝 生成的描述:")
        print("="*50)
        print(caption)
        print("="*50)
        print(f"💾 显存增量: {mem_used:.2f}GB")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        clear_gpu_memory()
        print("\n👋 程序结束")

if __name__ == "__main__":
    main()
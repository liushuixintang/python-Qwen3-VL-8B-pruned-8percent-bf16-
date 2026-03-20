import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import os
import time
import argparse
from pathlib import Path

# ====================== 强制GPU配置 ======================
# 检查CUDA可用性
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA不可用，无法强制使用GPU运行。请检查CUDA安装。")

# 设置CUDA设备
torch.cuda.set_device(0)  # 使用第一个GPU
device = torch.device("cuda:0")

print(f"🎮 强制使用GPU: {torch.cuda.get_device_name(0)}")
print(f"📊 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"🔢 CUDA版本: {torch.version.cuda}")

# ====================== 模型配置 ======================
MODEL_PATH = r"./Qwen3-VL-8B-pruned-8percent-bf16"  # 剪枝模型路径
DTYPE = torch.bfloat16  # 使用BF16格式减少显存占用
# ======================================================

def clear_gpu_memory():
    """清理GPU显存"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"🧹 GPU显存 - 已分配: {allocated:.2f}GB, 缓存: {cached:.2f}GB")

def load_model_gpu():
    """强制在GPU上加载模型 - 处理剪枝后的结构变化"""
    print("\n🚀 正在加载模型到GPU...")
    start_time = time.time()
    
    # 清理显存
    clear_gpu_memory()
    
    # 首先加载配置并修改以匹配剪枝后的结构
    from transformers import AutoConfig
    
    print("加载模型配置...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # 检查剪枝日志来确定新的维度
    import json
    pruning_log_path = os.path.join(MODEL_PATH, 'pruning_log.json')
    if os.path.exists(pruning_log_path):
        with open(pruning_log_path, 'r') as f:
            pruning_log = json.load(f)
        
        # 从第一层获取剪枝后的维度
        first_layer = pruning_log.get('layer_0', {})
        if first_layer and 'gate_proj' in first_layer:
            new_dim = first_layer['gate_proj']['new'][0]  # 11304
            print(f"从剪枝日志中获取到新的MLP维度: {new_dim}")
            
            # 修改配置中的中间层大小
            if hasattr(config, 'intermediate_size'):
                config.intermediate_size = new_dim
                print(f"更新config.intermediate_size为: {new_dim}")
    
    # 加载模型 - 使用修改后的配置
    print("加载模型权重（处理结构变化）...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        config=config,  # 使用修改后的配置
        torch_dtype=DTYPE,
        device_map=None,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,  # 关键：忽略大小不匹配的错误
    )
    
    # 强制移动到GPU
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    load_time = time.time() - start_time
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载完成！耗时: {load_time:.2f}秒")
    print(f"📊 模型参数量: {total_params/1e9:.2f}B")
    print(f"📍 模型设备: {next(model.parameters()).device}")
    print(f"🔤 数据类型: {next(model.parameters()).dtype}")
    
    # 验证模型结构
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
        first_layer = layers[0]
        if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'gate_proj'):
            actual_dim = first_layer.mlp.gate_proj.weight.shape[0]
            print(f"✅ 验证 - MLP gate_proj实际维度: {actual_dim}")
    
    # 显示显存使用
    clear_gpu_memory()
    
    return model, processor

def load_image(image_path):
    """加载并预处理图片"""
    try:
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                import requests
                from io import BytesIO
                response = requests.get(image_path, timeout=10)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
        else:
            image = image_path
            
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise Exception(f"图片加载失败 {image_path}: {e}")

def generate_caption(model, processor, image, prompt_template=None, max_new_tokens=256):
    """生成图像描述（反推提示词）"""
    
    # 默认提示词模板
    if prompt_template is None:
        prompt_template = "Describe this image in detail, including objects, scenes, colors, and any text visible."
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_template}
            ]
        }
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理输入
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    # 移动到GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 记录GPU显存使用
    before_mem = torch.cuda.memory_allocated() / 1024**3
    
    # 生成描述
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=DTYPE):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                repetition_penalty=1.0,
                length_penalty=1.0,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
    
    # 计算显存使用
    after_mem = torch.cuda.memory_allocated() / 1024**3
    
    # 解码输出
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return caption, after_mem - before_mem

def generate_detailed_prompt(model, processor, image, prompt_type="detailed"):
    """生成不同类型的提示词"""
    
    prompt_templates = {
        "simple": "What is in this image?",
        
        "detailed": "Describe this image in detail, including the main subject, background, colors, lighting, composition, and any text or people visible.",
        
        "style": "Describe the artistic style, medium, color palette, and visual techniques used in this image.",
        
        "objects": "List all objects, people, animals, and items visible in this image with their approximate locations.",
        
        "scene": "Describe the scene, setting, environment, atmosphere, and time of day in this image.",
        
        "action": "Describe the actions, interactions, poses, and expressions of subjects in this image.",
        
        "technical": "Provide a technical description of this image including composition, lighting, depth of field, camera angle, and focal point.",
        
        "stable_diffusion": "Write a detailed prompt suitable for Stable Diffusion or Midjourney that would generate this image. Include subject, style, lighting, composition, and quality tags.",
        
        "negative_prompt": "Based on this image, what elements or styles should be avoided in negative prompts? List undesirable elements."
    }
    
    if prompt_type not in prompt_templates:
        prompt_type = "detailed"
    
    prompt = prompt_templates[prompt_type]
    caption, mem_used = generate_caption(model, processor, image, prompt)
    
    return caption, mem_used

def verify_model_structure(model):
    """验证模型结构是否正确加载"""
    print("\n🔍 验证模型结构:")
    
    # 检查第一层的MLP维度
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
        first_layer = layers[0]
        
        if hasattr(first_layer, 'mlp'):
            mlp = first_layer.mlp
            
            if hasattr(mlp, 'gate_proj'):
                gate_dim = mlp.gate_proj.weight.shape[0]
                print(f"  gate_proj 维度: {gate_dim}")
            
            if hasattr(mlp, 'up_proj'):
                up_dim = mlp.up_proj.weight.shape[0]
                print(f"  up_proj 维度: {up_dim}")
            
            if hasattr(mlp, 'down_proj'):
                down_dim = mlp.down_proj.weight.shape[1]
                print(f"  down_proj 维度: {down_dim}")
    
    print("✅ 模型结构验证完成")

def batch_process_images(model, processor, image_folder, prompt_type="detailed", output_file="captions.txt"):
    """批量处理文件夹中的图片"""
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_paths = []
    
    # 收集所有图片
    folder = Path(image_folder)
    for ext in image_extensions:
        image_paths.extend(folder.glob(f'*{ext}'))
        image_paths.extend(folder.glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"在 {image_folder} 中没有找到图片")
        return
    
    print(f"\n📁 找到 {len(image_paths)} 张图片")
    
    results = []
    total_time = 0
    
    for i, img_path in enumerate(image_paths):
        print(f"\n--- 处理图片 {i+1}/{len(image_paths)}: {img_path.name} ---")
        
        try:
            # 加载图片
            image = load_image(str(img_path))
            
            # 计时
            start_time = time.time()
            
            # 生成描述
            caption, mem_used = generate_detailed_prompt(model, processor, image, prompt_type)
            
            # 计算时间
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # 保存结果
            result = {
                'image': str(img_path),
                'caption': caption,
                'time': elapsed,
                'memory': mem_used
            }
            results.append(result)
            
            # 打印结果
            print(f"📝 描述: {caption[:200]}..." if len(caption) > 200 else f"📝 描述: {caption}")
            print(f"⏱️  耗时: {elapsed:.2f}秒")
            print(f"💾 显存增量: {mem_used:.2f}GB")
            
            # 清理
            clear_gpu_memory()
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
    
    # 保存结果到文件
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"图片: {r['image']}\n")
                f.write(f"描述: {r['caption']}\n")
                f.write(f"耗时: {r['time']:.2f}秒\n")
                f.write("-" * 50 + "\n")
        
        print(f"\n✅ 结果已保存到: {output_file}")
        print(f"平均处理时间: {total_time/len(results):.2f}秒/张")

def interactive_mode(model, processor):
    """交互式反推模式"""
    print("\n🎯 进入交互式反推模式")
    print("可用的提示词类型:")
    for key in ["simple", "detailed", "style", "objects", "scene", "action", "technical", "stable_diffusion", "negative_prompt"]:
        print(f"  - {key}")
    
    current_image = None
    current_type = "detailed"
    
    while True:
        print("\n" + "-" * 50)
        print("命令:")
        print("  load <图片路径> - 加载图片")
        print("  type <提示词类型> - 切换提示词类型")
        print("  generate - 生成当前图片的描述")
        print("  info - 显示GPU信息")
        print("  verify - 验证模型结构")
        print("  quit - 退出")
        
        cmd = input("\n请输入命令: ").strip().lower()
        
        if cmd == 'quit':
            break
        
        elif cmd.startswith('load '):
            img_path = cmd[5:].strip()
            try:
                current_image = load_image(img_path)
                print(f"✅ 图片加载成功: {img_path}")
                print(f"📏 图片尺寸: {current_image.size}")
            except Exception as e:
                print(f"❌ 加载失败: {e}")
        
        elif cmd.startswith('type '):
            current_type = cmd[5:].strip()
            print(f"✅ 提示词类型已切换为: {current_type}")
        
        elif cmd == 'generate':
            if current_image is None:
                print("❌ 请先加载图片")
                continue
            
            try:
                print("🔄 生成中...")
                caption, mem_used = generate_detailed_prompt(model, processor, current_image, current_type)
                print(f"\n📝 生成的描述:\n{caption}")
                print(f"\n💾 显存使用: {mem_used:.2f}GB")
            except Exception as e:
                print(f"❌ 生成失败: {e}")
        
        elif cmd == 'info':
            print(f"\n📊 GPU信息:")
            print(f"  设备: {torch.cuda.get_device_name(0)}")
            print(f"  已分配显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"  缓存显存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            print(f"  可用显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.2f}GB")
        
        elif cmd == 'verify':
            verify_model_structure(model)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen3-VL 剪枝模型 - 图像反推提示词")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--folder", type=str, help="批量处理文件夹路径")
    parser.add_argument("--type", type=str, default="detailed", 
                       choices=["simple", "detailed", "style", "objects", "scene", "action", "technical", "stable_diffusion", "negative_prompt"],
                       help="提示词类型")
    parser.add_argument("--output", type=str, default="captions.txt", help="输出文件")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--verify", action="store_true", help="验证模型结构后退出")
    
    args = parser.parse_args()
    
    # 加载模型
    model, processor = load_model_gpu()
    
    # 验证模型结构
    verify_model_structure(model)
    
    if args.verify:
        return
    
    try:
        if args.interactive:
            interactive_mode(model, processor)
        
        elif args.image:
            # 单张图片处理
            print(f"\n📸 处理单张图片: {args.image}")
            image = load_image(args.image)
            caption, mem_used = generate_detailed_prompt(model, processor, image, args.type)
            
            print(f"\n📝 生成的{args.type}描述:")
            print("-" * 50)
            print(caption)
            print("-" * 50)
            print(f"💾 显存增量: {mem_used:.2f}GB")
        
        elif args.folder:
            # 批量处理
            batch_process_images(model, processor, args.folder, args.type, args.output)
        
        else:
            # 默认使用示例图片
            print("使用示例图片进行测试...")
            # 创建一个简单的测试图片
            image = Image.new('RGB', (512, 512), color='blue')
            caption, mem_used = generate_detailed_prompt(model, processor, image, "detailed")
            print(f"\n📝 生成的描述:\n{caption}")
            print(f"\n💾 显存使用: {mem_used:.2f}GB")
    
    finally:
        # 清理
        clear_gpu_memory()
        print("\n👋 程序结束")

if __name__ == "__main__":
    main()
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
import gc
import json
from tqdm import tqdm

# ====================== 配置 ======================
MODEL_PATH = r"D:\qwen3-8b\Qwen3-VL-8B-Instruct\qwen\Qwen3-VL-8B-Instruct"
SAVE_PATH = "./Qwen3-VL-8B-pruned-8percent-real"
PRUNE_RATE = 0.08
MAX_SHARD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes
# ==================================================

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

# 设置环境变量，避免CUDA问题
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用CUDA
torch.set_num_threads(8)  # 设置CPU线程数

print("🔥 正在 CPU 上加载模型...")
print("正在加载模型，这可能需要几分钟...")

# 强制在CPU上加载模型
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # CPU上使用float32
    low_cpu_mem_usage=True,
    device_map=None,  # 不使用自动设备映射
)
model = model.cpu()  # 确保在CPU上

print("✅ 模型加载完成")
print(f"模型设备: {next(model.parameters()).device}")

# 强制垃圾回收
gc.collect()

# 获取层
if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
    layers = model.model.language_model.layers
    print(f"找到 {len(layers)} 层 transformer 层")
else:
    raise ValueError("无法找到模型的层结构")

def prune_layer_mlp(layer, amount, layer_idx):
    """剪枝单层的MLP - 完全在CPU上执行"""
    if not hasattr(layer, 'mlp'):
        return layer, 0, {}
    
    mlp = layer.mlp
    pruned_params = 0
    layer_info = {}
    
    # 确保所有操作在CPU上
    with torch.no_grad():
        # 剪枝 gate_proj
        if hasattr(mlp, 'gate_proj') and mlp.gate_proj.weight is not None:
            # 确保权重在CPU上
            weight = mlp.gate_proj.weight.data.cpu()
            original_shape = weight.shape
            print(f"    gate_proj 原始形状: {original_shape}")
            
            # 计算重要性 - 使用float32避免精度问题
            importance = torch.norm(weight.float(), p=2, dim=1)
            keep_neurons = int(weight.size(0) * (1 - amount))
            
            if keep_neurons < weight.size(0) and keep_neurons > 0:
                # 找到最重要的神经元
                _, indices = torch.topk(importance, keep_neurons)
                indices = indices.sort()[0]
                
                # 应用剪枝
                new_weight = weight[indices, :].contiguous()
                mlp.gate_proj.weight.data = new_weight
                
                if mlp.gate_proj.bias is not None:
                    bias = mlp.gate_proj.bias.data.cpu()
                    mlp.gate_proj.bias.data = bias[indices].contiguous()
                
                pruned_params += (original_shape[0] - keep_neurons) * original_shape[1]
                layer_info['gate_proj'] = {
                    'original': list(original_shape),
                    'new': list(new_weight.shape)
                }
                print(f"    gate_proj 剪枝后: {new_weight.shape}")
            else:
                print(f"    gate_proj 保持原状")
        
        # 剪枝 up_proj
        if hasattr(mlp, 'up_proj') and mlp.up_proj.weight is not None:
            weight = mlp.up_proj.weight.data.cpu()
            original_shape = weight.shape
            print(f"    up_proj 原始形状: {original_shape}")
            
            importance = torch.norm(weight.float(), p=2, dim=1)
            keep_neurons = int(weight.size(0) * (1 - amount))
            
            if keep_neurons < weight.size(0) and keep_neurons > 0:
                _, indices = torch.topk(importance, keep_neurons)
                indices = indices.sort()[0]
                
                new_weight = weight[indices, :].contiguous()
                mlp.up_proj.weight.data = new_weight
                
                if mlp.up_proj.bias is not None:
                    bias = mlp.up_proj.bias.data.cpu()
                    mlp.up_proj.bias.data = bias[indices].contiguous()
                
                pruned_params += (original_shape[0] - keep_neurons) * original_shape[1]
                layer_info['up_proj'] = {
                    'original': list(original_shape),
                    'new': list(new_weight.shape)
                }
                print(f"    up_proj 剪枝后: {new_weight.shape}")
            else:
                print(f"    up_proj 保持原状")
        
        # 剪枝 down_proj
        if hasattr(mlp, 'down_proj') and mlp.down_proj.weight is not None:
            weight = mlp.down_proj.weight.data.cpu()
            original_shape = weight.shape
            print(f"    down_proj 原始形状: {original_shape}")
            
            importance = torch.norm(weight.float(), p=2, dim=0)
            keep_neurons = int(weight.size(1) * (1 - amount))
            
            if keep_neurons < weight.size(1) and keep_neurons > 0:
                _, indices = torch.topk(importance, keep_neurons)
                indices = indices.sort()[0]
                
                new_weight = weight[:, indices].contiguous()
                mlp.down_proj.weight.data = new_weight
                # down_proj的bias通常对应输出维度，不需要剪
                
                pruned_params += weight.size(0) * (original_shape[1] - keep_neurons)
                layer_info['down_proj'] = {
                    'original': list(original_shape),
                    'new': list(new_weight.shape)
                }
                print(f"    down_proj 剪枝后: {new_weight.shape}")
            else:
                print(f"    down_proj 保持原状")
    
    return layer, pruned_params, layer_info

# 逐层剪枝
total_pruned = 0
pruning_log = {}

print(f"\n开始剪枝，剪枝率: {PRUNE_RATE*100}%")
print("=" * 50)

for i, layer in enumerate(layers):
    print(f"\n处理第 {i+1}/{len(layers)} 层...")
    
    try:
        layer, pruned, info = prune_layer_mlp(layer, PRUNE_RATE, i)
        total_pruned += pruned
        if info:
            pruning_log[f'layer_{i}'] = info
        print(f"  本层剪枝参数: {pruned/1e6:.2f}M")
    except Exception as e:
        print(f"  处理第 {i+1} 层时出错: {e}")
        continue
    
    # 每层都强制垃圾回收
    gc.collect()
    
    # 每5层打印一次进度总结
    if (i + 1) % 5 == 0:
        print(f"\n--- 已完成 {i+1}/{len(layers)} 层，累计剪枝: {total_pruned/1e6:.2f}M ---")

print(f"\n✅ 剪枝完成！")
print(f"总剪枝参数量: {total_pruned/1e6:.2f}M")
print(f"总剪枝参数量: {total_pruned/1e9:.2f}B")

# 保存剪枝日志
with open(os.path.join(SAVE_PATH, 'pruning_log.json'), 'w', encoding='utf-8') as f:
    json.dump(pruning_log, f, indent=2, ensure_ascii=False)

# 统计剪枝后的总参数量
total_params_after = sum(p.numel() for p in model.parameters())
print(f"\n剪枝后模型总参数量: {total_params_after/1e9:.2f}B")

print("\n💾 正在准备保存模型...")

# 首先保存配置和处理器
print("保存配置...")
model.config.save_pretrained(SAVE_PATH)

print("保存处理器...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.save_pretrained(SAVE_PATH)

# 获取状态字典并计算分片
print("获取模型权重...")
state_dict = model.state_dict()

# 计算总分片数
total_size = 0
for param in state_dict.values():
    total_size += param.numel() * param.element_size()

num_shards = (total_size + MAX_SHARD_SIZE - 1) // MAX_SHARD_SIZE
print(f"模型总大小: {total_size / 1024**3:.2f}GB")
print(f"分片数量: {num_shards}")

# 逐片保存
print("\n开始保存分片...")
shard_size = 0
shard_num = 1
current_shard = {}
shard_files = []

for name, param in tqdm(state_dict.items(), desc="保存权重"):
    param_size = param.numel() * param.element_size()
    
    if shard_size + param_size > MAX_SHARD_SIZE and current_shard:
        shard_filename = f"pytorch_model-{shard_num:05d}-of-{num_shards:05d}.bin"
        shard_path = os.path.join(SAVE_PATH, shard_filename)
        
        print(f"\n保存分片 {shard_num}/{num_shards}...")
        torch.save(current_shard, shard_path)
        shard_files.append(shard_filename)
        
        shard_num += 1
        shard_size = 0
        current_shard = {}
        gc.collect()
    
    current_shard[name] = param.cpu()
    shard_size += param_size

# 保存最后一个分片
if current_shard:
    shard_filename = f"pytorch_model-{shard_num:05d}-of-{num_shards:05d}.bin"
    shard_path = os.path.join(SAVE_PATH, shard_filename)
    print(f"\n保存分片 {shard_num}/{num_shards}...")
    torch.save(current_shard, shard_path)
    shard_files.append(shard_filename)

# 创建index文件
index_dict = {
    "metadata": {
        "total_size": total_size,
        "prune_rate": PRUNE_RATE,
        "pruned_params": total_pruned
    },
    "weight_map": {}
}

# 为每个参数记录它在哪个分片
for i, name in enumerate(state_dict.keys()):
    shard_idx = i // (len(state_dict) // len(shard_files) + 1)
    index_dict["weight_map"][name] = shard_files[min(shard_idx, len(shard_files)-1)]

# 保存index文件
index_path = os.path.join(SAVE_PATH, "pytorch_model.bin.index.json")
with open(index_path, 'w', encoding='utf-8') as f:
    json.dump(index_dict, f, indent=2)

print(f"\n🎉 模型保存完成！")
print(f"保存路径: {SAVE_PATH}")
print(f"分片文件: {shard_files}")

# 创建README说明
readme_content = "# Qwen3-VL-8B 剪枝模型\n\n"
readme_content += f"- 剪枝率: {PRUNE_RATE*100}%\n"
readme_content += f"- 原始参数量: 8.77B\n"
readme_content += f"- 剪枝后参数量: {total_params_after/1e9:.2f}B\n"
readme_content += f"- 模型大小: {total_size/1024**3:.2f}GB (float32格式)\n"
readme_content += f"- 如果转换为float16，大小约为: {total_size/1024**3/2:.2f}GB\n\n"
readme_content += "## 使用方法\n```python\n"
readme_content += "from transformers import AutoModelForImageTextToText, AutoProcessor\n\n"
readme_content += f'model = AutoModelForImageTextToText.from_pretrained(r"{SAVE_PATH}")\n'
readme_content += f'processor = AutoProcessor.from_pretrained(r"{SAVE_PATH}")\n'
readme_content += "```\n"

with open(os.path.join(SAVE_PATH, "README.md"), 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("\n📝 README文件已创建")
print("\n✨ 所有步骤完成！")
from modelscope import snapshot_download




model_dir = snapshot_download(
    model_id="qwen/Qwen3-VL-8B-Instruct",  # ✅ 基础版
    local_dir=r"H:\新建文件夹",
)





print("✅ 下载完成！模型保存在：", model_dir)
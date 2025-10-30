from ultralytics import YOLO
import json
import numpy as np
from collections import defaultdict
import copy


STD_THRESHOLD = 1.0  # 标准差阈值（只保留>1的层）

# Load a model
origin_model = YOLO("yolo11x.pt")  # load an official model

metadata = {
    "stride": int(max(origin_model.stride)),
    "task": origin_model.task,
    "batch": 1,
    "imgsz": 640,
    "names": origin_model.names,
    "channels": origin_model.yaml.get("channels", 3),
}  # model metadata 自己构建engine的话要写到engine中


# Export the model
# model.export(format="engine",batch=1, int8=True, data='/workspace/yolov11/coco/coco.yaml', fraction=0.01) # 官方导出这一行就够了
############################################# mtq part ###################################################
import os
from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
import tensorrt as trt
from pathlib import Path
def build_calibration_dataset(data_dir, max_samples=100):
    """
    构造用于量化校准的图像张量列表
    :param data_dir: 图像目录路径
    :param max_samples: 最大样本数
    :return: List[Tensor]，每个张量形状为 [1, 3, 640, 640]
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),  # 转为 [0,1]，形状 [C,H,W]
    ])

    calibration_data = []
    image_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".jpg")
    ])[:max_samples]

    for fname in image_files:
        path = os.path.join(data_dir, fname)
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # 添加 batch 维度 → [1, 3, 256, 256]
        calibration_data.append(tensor)

    return calibration_data

if __name__ == '__main__':
    logger = trt.Logger(trt.Logger.INFO)
    # if verbose:
    logger.min_severity = trt.Logger.Severity.VERBOSE
    
    builder = trt.Builder(logger) # 创建builder
    config = builder.create_builder_config() # 创建builder config
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # 使用显示batch创建网络
    network = builder.create_network(flag)
    
    parser = trt.OnnxParser(network, logger) # ONNX模型加载到TensorRT的INetworkDefinition中
    
    # load model and export
    ##################################### mtq #############################################
    model = origin_model.model #  调用yolo加载的torch模型
    model.eval()
    state_dict = model.state_dict()
    example_inputs = torch.randn(1, 3, 640, 640)
    import modelopt.torch.quantization as mtq
    # 验证模型参数是否存在
    if not state_dict:
        raise ValueError("模型未加载到参数，请检查路径是否正确！")

    # -------------------------- 按层分组并计算标准差 --------------------------
    layer_params = defaultdict(list)

    # 分组参数（按层名）
    for param_name, param_tensor in state_dict.items():
        if "." in param_name and param_name.split(".")[-1] in ("weight", "bias"):
            layer_name = ".".join(param_name.split(".")[:-1])  # 提取层名（去掉.weight/.bias）
            layer_params[layer_name].append(param_tensor.detach().cpu().numpy())

    # 计算每层标准差
    layer_stats = []
    for layer_name, param_list in layer_params.items():
        merged_params = np.concatenate([p.flatten() for p in param_list])
        std = np.std(merged_params)
        total_params = merged_params.size
        layer_stats.append({
            "layer_name": layer_name,
            "std": std,
            "total_params": total_params
        })

    # -------------------------- 筛选标准差>1的层并生成CUSTOM_CFG --------------------------
    # 筛选目标层（标准差>1）
    filtered_layers = [stats for stats in layer_stats if stats["std"] > STD_THRESHOLD]
    CUSTOM_CFG = copy.deepcopy(mtq.INT8_DEFAULT_CFG)  # 修改默认INT8量化的配置

    for stats in filtered_layers:
        original_layer_name = stats["layer_name"]
        # 处理层名：去掉前缀"model."（如"model.model.23.dfl.conv" → "model.23.dfl.conv"）
        # processed_layer = original_layer_name.replace("model.", "", 1)  # 只去掉第一个"model."
        # 用*包裹处理后的层名（确保匹配）
        config_key = f"*{original_layer_name}*"
        # config_key = f"*{processed_layer}*"
        CUSTOM_CFG["quant_cfg"][config_key] = {"enable": False}

    # -------------------------- 输出结果 --------------------------
    print("=== 标准差 > 1 的层及其配置 ===")
    print("筛选出的层（原始层名, 标准差）：")
    for stats in filtered_layers:
        print(f"- {stats['layer_name']}: {stats['std']:.6f}")

    print("\n自动生成的CUSTOM_CFG：")
    import pprint
    pprint.pprint(CUSTOM_CFG)
    
    # --------------------------  mtq量化  ------------------------------ 
    data_loader = build_calibration_dataset('/workspace/yolov11/coco/images/train2017', max_samples=500) # 路径指向训练集 max_samples 校验集大小
    def forward_loop(model):
        for batch in data_loader:
            model(batch)
    
    model = mtq.quantize(model, CUSTOM_CFG, forward_loop) # mtq量化

    mtq.print_quant_summary(model)
    
    ##################################  export onnx   ###############################################
    onnx_file = 'mtq_int8_layer.onnx'
    torch.onnx.export(model, example_inputs, onnx_file,input_names=["images"],dynamic_axes=None)  
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")
    
    ##################################  export engine   ###############################################
    config.set_flag(trt.BuilderFlag.INT8)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED # 设置精度并打印构建信息
    
    build = builder.build_serialized_network  # 构建engine
    engine_file = 'mtq_int8_layer.engine' # 导出engine的路径+名字
    with build(network, config) as engine, open(engine_file, "wb") as t:
        if metadata is not None:# 如果有元数据 yolo依靠这个元数据运行
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
        # Model
        t.write(engine) # 写入

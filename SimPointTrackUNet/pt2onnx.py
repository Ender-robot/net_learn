import torch
import onnx
import onnxsim
from torch.export import Dim
from model.SimPointTrackUNet import SimPointTrackUNet

def main():
    # 配置
    config = {
        "weight_path" : r"D:\Admin-Ender\net_learn\SimPointTrackUNet\log\2V3A95L0002.pt",
        "device"      : "0",
        "onnx_file"   : r"D:\Admin-Ender\net_learn\SimPointTrackUNet\log\2V3A95L0002.onnx",
        "opset_ver"   : 18,
        "simplified"  : True,
    }

    # 选择设备
    if torch.cuda.is_available() and config["device"] != "cpu":
        config["device"] = torch.device(f"cuda:{config['device']}")
        print(f"Used CUDA: {config['device']}")
    else:
        config["device"] = torch.device("cpu")
        print(f"Used CPU")

    dummy_input = torch.randn(1, 15, 2).to(config['device'])

    pt2onnx(dummy_input, config)

def pt2onnx(dummy_input, config : dict):
    # 读取参数
    weight_path = config["weight_path"]
    device      = config["device"]
    onnx_file   = config["onnx_file"]
    opset_ver   = config["opset_ver"]
    simplified  = config["simplified"]

    # 加载模型
    model = SimPointTrackUNet()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # 导出
    batch_dim = Dim("batch_size") # 创建动态轴

    torch.onnx.export(
        model, # 要转换的模型
        dummy_input, # 虚拟输入
        onnx_file, # 输出的ONNX路径
        export_params=True, # 导出训练好的参数
        opset_version=opset_ver, # ONNX 算子集版本
        do_constant_folding=True, # 是否执行常量折叠优化
        input_names=['input'],   # 输入节点的名称
        output_names=['output'], # 输出节点的名称
        dynamic_shapes={'x': {0: batch_dim}} # 动态轴
        )
    
    print("导出完成 !")
    
    # 简化
    if simplified:
        onnx_model = onnx.load(onnx_file)
        model_simplified, check = onnxsim.simplify(onnx_model)

        if not check:
            print("简化失败 !")
            return
        
        onnx.save(model_simplified, onnx_file)
        print("简化成功 !")

if __name__ == "__main__":
    main()

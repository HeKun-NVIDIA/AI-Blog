# Torch-TensorRT 2.4 Windows 支持中的 C++ 运行时支持、转换器中增强的动态形状支持

Torch-TensorRT 2.4.0 面向 PyTorch 2.4、CUDA 12.4（可通过 PyTorch 包索引获取 CUDA 11.8/12.1 的版本 - https://download.pytorch.org/whl/cu118 https://download.pytorch.org/whl/cu121）和 TensorRT 10.1。
此版本引入了对 Windows 平台上 C++ 运行时的官方支持，但仅限于 dynamo 前端，支持 AOT 和 JIT 工作流。用户现在可以在 Windows 上使用 Python 和 C++ 运行时。此外，此版本扩展了支持范围，包括所有 Aten Core 运算符（torch.nonzero 除外），并显著增加了更多转换器的动态形状支持。此版本首次支持 Python 3.12。

## 完全支持 Windows
在此版本中，我们在 Windows 中引入了 C++ 和 Python 运行时支持。用户现在可以直接在 Windows 上使用 TensorRT 优化 PyTorch 模型，无需更改代码。C++ 运行时是默认选项，用户可以通过指定 `use_python_runtime=True` 来启用 Python 运行时

```Python
import torch
import torch_tensorrt
import torchvision.models as models

model = models.resnet18(pretrained=True).eval().to("cuda")
input = torch.randn((1, 3, 224, 224)).to("cuda")
trt_mod = torch_tensorrt.compile(model, ir="dynamo", inputs=[input])
trt_mod(input)
```

## 转换器中增强的 Op 支持
转换器的支持率接近核心 ATen 的 100%。此时，回退到 PyTorch 执行要么是由于转换器的特定限制，要么是由于用户编译器设置的某种组合（例如 torch_executed_ops、动态形状）。此版本还扩展了支持动态形状的运算符数量。dryrun 将提供有关您的模型 + 设置支持的具体信息。


















































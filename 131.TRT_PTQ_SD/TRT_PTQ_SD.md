#! https://zhuanlan.zhihu.com/p/686775075
# 利用TensorRT的8位PTQ将Stable Diffusion速度提高 2 倍

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/stable-diffusion-sdxl-featured.png)

在生成人工智能的动态领域中，扩散模型脱颖而出，成为生成带有文本提示的高质量图像的最强大的架构。 像稳定扩散这样的模型已经彻底改变了创意应用。

然而，由于需要迭代去噪步骤，扩散模型的推理过程可能需要大量计算。 这对于努力实现最佳端到端推理速度的公司和开发人员提出了重大挑战。

从 [NVIDIA TensorRT 9.2.0](https://developer.nvidia.com/tensorrt-getting-started) 开始，我们开发了一流的量化工具包，具有改进的 8 位（FP8 或 INT8）训练后量化 (PTQ: Post-Training Quantization)，可显着加快 NVIDIA 硬件上的扩散部署，同时保持图像质量 。 TensorRT 的 8 位量化功能已成为许多生成型 AI 公司的首选解决方案，特别是创意视频编辑应用程序的领先提供商。

在这篇文章中，我们讨论 TensorRT 与 Stable Diffusion XL 的性能。 我们介绍了使 TensorRT 成为低延迟稳定扩散推理的首选的技术优势。 最后，我们演示如何使用 TensorRT 通过几行更改来加速模型。

## 性能指标
与在 FP16 中运行的本机 PyTorch 的 `torch.compile` 相比，用于扩散模型的 NVIDIA TensorRT INT8 和 FP8 量化方案在 NVIDIA RTX 6000 Ada GPU 上实现了 1.72 倍和 1.95 倍的加速。 FP8 相对于 INT8 的额外加速主要归因于多头注意力 (MHA) 层的量化。 使用 TensorRT 8 位量化可以增强生成式 AI 应用程序的响应能力并降低推理成本。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/benchmark-inference-speedup-int8-fp8-1.png)

除了加速推理之外，TensorRT 8 位量化还擅长保持图像质量。 通过专有的量化技术，它生成与原始 FP16 图像非常相似的图像。 我们将在本文后面介绍这些技术。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/TensorRT-INT8-and-FP8-image-quality-comared-to-FP16.png)


## TensorRT 解决方案：克服推理速度挑战
尽管 PTQ 被认为是减少内存占用并加速许多 AI 任务推理的首选压缩方法，但它在扩散模型上并不能开箱即用。 扩散模型具有独特的多时间步去噪过程，并且噪声估计网络在每个时间步的输出分布可能会有很大变化。 这使得简单的 PTQ 校准方法不适用。

在现有技术中，[SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf) 作为一种流行的 PTQ 方法脱颖而出，可为 LLM 实现 8 位权重、8 位激活 (W8A8) 量化。 其主要创新在于解决激活异常值的方法，通过数学上等效的变换将量化挑战从激活转移到权重。

尽管它很有效，但用户在 SmoothQuant 中手动定义参数时经常遇到困难。 实证研究还表明，SmoothQuant 难以适应不同的图像特征，限制了其在现实场景中的灵活性和性能。 此外，其他现有的扩散模型量化技术仅针对单个版本的扩散模型量身定制，而用户正在寻找一种可以加速各种版本模型的通用方法。

为了应对这些挑战，NVIDIA TensorRT 开发了复杂的细粒度调整管道，以确定 SmoothQuant 模型每一层的最佳参数设置。 您可以根据特征图的具体特征开发自己的调整管道。 与基于客户需求的现有方法相比，此功能使 TensorRT 量化能够获得卓越的图像质量，保留原始图像的丰富细节。

根据 Q-Diffusion 的研究结果，激活分布在不同的时间步长内可能会有很大差异，并且图像的形状和整体风格主要在去噪过程的初始阶段确定。 因此，使用传统的最大校准会导致初始步骤中出现较大的量化误差。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/activation-distribution-high-low-noise-range-1.png)

相反，我们有选择地使用所选步骤范围中的最小量化缩放因子，因为我们发现激活中的异常值对最终图像质量并不那么重要。 这种量身定制的方法，我们将其命名为“百分比定量”，重点关注步长范围的重要百分位。 它使 TensorRT 能够生成与原始 FP16 精度生成的图像几乎相同的图像。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/percentile-quant-image-generation-int8-compared-fp16-1.png)

## 使用 TensorRT 8 位量化加速扩散模型
[/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/9.3/demo/Diffusion#faster-text-to-image-using-sdxl--int8-quantization-using-ammo) GitHub 存储库现在托管端到端、SDXL、8 位推理管道，提供即用型解决方案以在 NVIDIA GPU 上实现优化的推理速度。

运行单个命令即可使用 Percentile Quant 生成图像，并使用 demoDiffusion 测量延迟。 在本节中，我们使用 INT8 作为示例，但 FP8 的工作流程基本相同。

```bash
python demo_txt2img_xl.py "enchanted winter forest with soft diffuse light on a snow-filled day" --version xl-1.0 --onnx-dir onnx-sdxl --engine-dir engine-sdxl --int8 --quantization-level 3
```

以下是该命令所涉及的主要步骤的概述：

* 校准
* 导出 ONNX
* 构建 TensorRT 引擎

### 校准
校准是量化过程中计算目标精度范围的步骤。 目前，TensorRT 中的量化功能封装在 nvidia-ammo 中，该依赖项已包含在 TensorRT 8 位量化示例中。

```python
# Load the SDXL-1.0 base model from HuggingFace
import torch
from diffusers import DiffusionPipeline
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
base.to("cuda")
 
# Load calibration prompts:
from utils import load_calib_prompts
cali_prompts = load_calib_prompts(batch_size=2,prompts="./calib_prompts.txt")
 
# Create the int8 quantization recipe
from utils import get_percentilequant_config
quant_config = get_percentilequant_config(base.unet, quant_level=3.0, percentile=1.0, alpha=0.8)
 
# Apply the quantization recipe and run calibration  
import ammo.torch.quantization as atq 
quantized_model = atq.quantize(base.unet, quant_config, forward_loop)
 
# Save the quantized model
import ammo.torch.opt as ato
ato.save(quantized_model, 'base.unet.int8.pt')
```

### 导出 ONNX
获得量化模型检查点后，您可以导出 ONNX 模型。

```python
# Prepare the onnx export  
from utils import filter_func, quantize_lvl
base.unet = ato.restore(base.unet, 'base.unet.int8.pt')
quantize_lvl(base.unet, quant_level=3.0)
atq.disable_quantizer(base.unet, filter_func) # `filter_func` is used to exclude layers you don't quantize
  
# Export the ONNX model
from onnx_utils import ammo_export_sd
base.unet.to(torch.float32).to("cpu")
ammo_export_sd(base, 'onnx_dir', 'stabilityai/stable-diffusion-xl-base-1.0')
```

### 构建 TensorRT 引擎
使用 INT8 UNet ONNX 模型，您可以构建 TensorRT 引擎。

```bash
trtexec --onnx=./onnx_dir/unet.onnx --shapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=unetxl.trt.plan
```

## 总结
在生成式人工智能时代，拥有优先考虑易用性的推理解决方案至关重要。 借助 NVIDIA TensorRT，您可以通过其专有的 8 位量化技术无缝实现高达 2 倍的推理速度加速，同时确保图像质量不受影响，从而实现卓越的用户体验。

TensorRT 对平衡速度和质量的承诺凸显了其作为加速 AI 应用程序的领先选择的地位，使您能够轻松交付尖端解决方案。

我将在 NVIDIA GTC 大会期间为大家带来免费中文在线解读：
NVIDIA CUDA 最新特性以及生成式 AI 相关内容，包括 Stable Diffusion 模型部署实践，以及介绍用于视觉内容生成的 Edify 模型，点击链接了解详情并注册参会：

https://www.nvidia.cn/gtc-global/session-catalog/?search=WP62435%20WP62832%20WP62400&ncid=ref-dev-945313#/
























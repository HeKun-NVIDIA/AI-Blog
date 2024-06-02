# 使用 GPU 加速的 nvImageCodec 推进医学图像解码

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/image2.png)


本文深入探讨了使用 nvJPEG2000 库在 AWS HealthImaging 中解码 DICOM 医学图像的功能。 我们将引导您了解图像解码的复杂性，向您介绍 AWS HealthImaging，并探索 GPU 加速解码解决方案带来的进步。

通过 GPU 加速的 nvJPEG2000 库开始在 AWS HealthImaging 中提高吞吐量并降低解读医学图像的成本，这代表着云环境中运营效率的重大进步。 这些创新有望节省大量成本，预计此类工作负载的潜在成本削减总计达数亿美元。

## JPEG 2000
正确实施 JPEG 2000 会带来极大的复杂性，早期遇到的互操作性问题阻碍了不同系统之间的无缝集成。 这种复杂性成为广泛采用的障碍。 然而，高吞吐量 JPEG 2000 (HTJ2K) 编码系统的推出标志着图像压缩技术的显着进步。 JPEG 2000 标准第 15 部分中概述的 HTJ2K 通过用更高效的 FBCOT（优化截断快速块编码）替换原始块编码算法 EBCOT（优化截断嵌入式块编码）来提高吞吐量。

这一新标准解决了解码速度限制，并为 JPEG 2000 在医学成像领域的更广泛采用打开了大门。 HTJ2K 支持无损和有损压缩，在保留关键医疗细节和实现高效存储之间提供平衡。 具有任意宽度和高度的灰度和彩色图像，以及每通道高达 16 位的支持，展示了 HTJ2K 的适应性。 新标准对分解级别没有限制，提供了广泛的选择。


## nvJPEG2000 库
随着通过nvJPEG2000等库进行GPU加速的出现，HTJ2K的解码性能达到了新的高度。 这一进步释放了 JPEG 2000 在医学图像处理方面的真正潜力，使其成为医疗保健提供者、研究人员和开发人员可行且高效的解决方案。 nvJPEG2000 提供 C API，包括用于解码单个图像的 nvjpeg2kDecode 和用于解码图像中特定图块的 nvjpeg2kDecodeTile 等函数。 图书馆提供：

统一API接口nvImageCodec：开源库与Python无缝集成，为开发人员提供了便捷的接口。
解码性能分析：HTJ2K 与传统 JPEG 2000 之间的解码性能对比分析，包括对 GPU 加速的见解。
为了确保可用性、高性能和生产就绪性，本文探讨了 HTJ2K 解码与 MONAI（专为医学图像分析而设计的框架）的集成。 MONAI Deploy App SDK 提供高性能功能，并促进医学成像 AI 应用程序的调试。 它还深入研究了使用 AWS HealthImaging、MONAI 和 nvJPEG2000 进行医学图像处理的相关成本优势。

使用 AWS HealthImaging 进行企业级医疗图像存储
得益于无损 HTJ2K 编码和 AWS 高性能网络骨干，AWS HealthImaging 可以从任何地方进行亚秒级图像检索，并快速访问存储在云中的图像。 它被设计为与工作流程无关，无缝集成到现有的医学成像工作流程中。 它也是符合 DICOM 标准的解决方案，可确保互操作性并遵守医学图像通信的行业标准。 该服务提供原生 API，用于可扩展和快速的图像摄取，适应不断增长的医学成像数据量。


## GPU 加速图像解码
为了进一步增强图像解码性能，AWS HealthImaging 支持 GPU 加速，特别是利用 NVIDIA nvJPEG2000 库。 这种 GPU 加速可确保快速高效地解码医学图像，使医疗保健提供者能够以前所未有的速度访问关键信息。 HTJ2K 解码支持的功能包含多种选项，可适应不同的图像类型、尺寸、压缩需求和解码场景，使其成为各种图像处理应用的多功能且适应性强的选择。 其中包括以下内容：

* 图像类型：HTJ2K兼容任意宽度和高度的灰度和彩色图像。 它适应不同的图像格式和尺寸。
* 位深度：HTJ2K 处理位深度高达每通道 16 位的图像。 这种高位深度支持可确保颜色和细节的准确表示。
* 无损压缩：HTJ2K 标准支持无损压缩，可以在不丢失任何数据的情况下保留图像质量。
* 统一的代码块配置：HTJ2K内的所有代码块均符合HT（高吞吐量）标准。 没有使用细化代码块，简化了解码过程。
* 码块大小：HTJ2K采用不同的码块大小，包括64×64、32×32和16×16。 这种适应性使得能够有效地表示具有不同细节和复杂程度的图像。
* 级数顺序：HTJ2K 支持多种级数顺序，包括下列顺序。 这些级数顺序为图像数据的组织和传输方式提供了灵活性。
    * LRCP（层-分辨率-组件-位置）
    * RLCP（分辨率层组件位置）
    * RPCL（分辨率-位置-组件-层）
    * PCRL（位置组件分辨率层）
    * CPRL（组件-位置-分辨率-层）
*可变分解级别：该标准允许各种数量的分解级别，范围从 1 到 5。这种分解灵活性提供了根据特定要求优化图像压缩的选项。
* 具有不同分块大小的多分块解码：HTJ2K 能够解码分为多个分块的图像，并且这些分块可以具有不同的大小。 此功能通过在更小的、可管理的部分中处理大图像，增强了有效解码大图像的能力。

## AWS HealthImaging 
在本演练中，我们展示了 AWS HealthImaging 的使用。 我们演示了利用 GPU 加速接口使用 SageMaker 多模型端点进行图像解码的过程。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/aws-network-backbone-interface.png)


### 第 1 步：暂存您的 DICOM 图像
首先将您的 DICOM 图像暂存在 Amazon S3 存储桶中。 AWS HealthImaging 与推出的合作伙伴产品集成，提供最适合您的工作流程的各种工具，并在指定的 S3 存储桶中上传和组织您的 DICOM 图像数据。 您还可以探索 AWS 开放数据计划。 公共 S3 存储桶中存在包含合成医学成像数据的开放数据集，例如 Synthea Coherent。

### 步骤2：调用DICOM数据导入API
将 DICOM 图像暂存在 S3 存储桶后，下一步涉及调用本机 API 将 DICOM 数据导入 AWS HealthImaging。 此托管 API 促进了平稳且自动化的流程，确保您的医学图像数据得到有效传输并为进一步优化做好准备。

### 步骤 3：在数据湖中索引 DICOM 标头
成功导入后，从 AWS HealthImaging 检索 DICOM 标头、解压缩数据 blob，并将这些 JSON 对象写入数据湖 S3 存储桶。 从那里，您可以利用 AWS 数据湖分析工具，例如 Amazon Glue 生成数据目录、Amazon Athena 执行即席 SQL 查询以及 Amazon QuickSight 构建数据可视化仪表板。 您还可以将图像元数据与其他健康数据模式相结合来执行多模式数据分析。

### 第 4 步：访问您的医学图像数据
借助托管 API，访问将成为一种无缝体验。 AWS HealthImaging 以亚秒级的速度提供对成像数据的高性能和精细访问。

在云上构建 PACS 查看器和 VNA 解决方案的 AWS 合作伙伴可以将其图像查看应用程序与 AWS HealthImaging 集成。 这些应用程序经过优化，可为大规模查看和分析医学图像提供用户友好且高效的体验。 AWS 合作伙伴 PACS 的示例包括 Allina Health 案例研究、Visage Imaging 和 Visage AWS。

科学家和研究人员可以利用 Amazon SageMaker 执行 AI 和 ML 建模，以解锁高级见解并自动执行审阅和注释任务。 Amazon Sagemaker 可与 MONAI 结合使用来开发强大的 AI 模型。 使用 Amazon SageMaker 笔记本，用户可以从 AWS HealthImaing 检索像素帧，使用 itkwidget 等开源工具可视化医学图像，并创建 SageMaker 托管训练作业或模型托管终端节点。

作为一项符合 HIPAA 要求的服务，AWS HealthImaging 可以灵活地向远程用户授予和审核对医疗图像数据的安全访问权限。 访问控制由 Amazon Identity and Access Management 管理，确保授权用户可以对 ImageSet 数据进行精细访问。 Amazon CloudTrail 还可以监控访问活动，以跟踪谁在什么时间访问了哪些数据。

### 步骤5：支持GPU的HTJ2K解码
在典型的 AI 或 ML 工作流程（CPU 解码路径）中，HTJ2K 编码的像素帧将加载到 CPU 内存中，然后在 CPU 中解码并转换为张量。 这些可以由 GPU 复制和处理。 nvJPEG2000 可以从 AWS HealthImaging 获取编码像素并将其直接解码到 GPU 内存中（GPU 解码路径），MONAI 具有内置函数，可将图像数据转换为张量，以供深度学习模型使用。 与 CPU 解码方法相比，它的路径更短，如下图所示。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/aws-healthimaging-api-interface.png)


此外，nvJPEG2000 的 GPU 加速显着提高了解码性能，减少了延迟并增强了整体响应能力。 该库与Python无缝集成，为开发人员提供了熟悉且强大的图像解码任务环境。

在 [Amazon SageMaker ](https://aws.amazon.com/sagemaker/)上运行的[演示notebook](https://github.com/aws-samples/monai-on-aws-workshop/)展示了如何以可扩展且高效的方式集成和利用 GPU 加速图像解码的强大功能。 在我们的实验中，SageMaker g4dn.2xlarge 实例上的 GPU 解码速度比 SageMaker m5.2xlarge 实例上的 CPU 解码速度快约 7 倍（下图）。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/decoding-images-gpu-cpu-acceleration-comparison.png)

在本实验中，我们使用了 Synthea Coherent 数据集中的合成大脑 MRI 图像。 GPU 加速对于不同大小的数据集表现出类似的加速因子。 上面标记的图像集包含脑 MRI 和像素帧。 这些像素帧代表 DICOM MRI 图像，并以压缩的 HTJ2K 数据格式进行编码。

## 成本效益分析
结合先进的图像解码技术，AWS HealthImaging 不仅提高了效率，还为医疗保健组织提供了经济高效的解决方案。 所提出的解决方案的端到端成本效益是巨大的，特别是考虑到通过 GPU 加速实现的令人印象深刻的吞吐量加速。

EC2 G4 实例上的单个 NVIDIA T4 GPU 的加速比 CPU 基准提高了大约 5 倍，而 EC2 G6 实例上的新 L4 GPU 上的这一改进进一步增强到了令人印象深刻的 12 倍。 通过多个 GPU 实例进行扩展，性能表现出近乎线性的可扩展性，在四个 NVIDIA T4 GPU 和四个 NVIDIA L4 GPU 上分别达到约 19 倍和 48 倍。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/image-decoding-speedups-various-gpus-2.png)


在解码性能方面，我们与OpenJPEG进行了对比分析。 对于 CT1 16 位 512×512 灰度图像，我们注意到不同 GPU 配置的速度显着提高了 2.5 倍。 此外，对于尺寸为 3064×4774 的较大 MG1 16 位灰度图像，我们在各种 GPU 设置上实现了令人印象深刻的 8 倍速度提升。

为了全面评估年度云成本和能源使用情况，我们的计算基于标准分段工作负载。 此工作负载涉及每分钟将 500 个 DICOM 文件上传到 MONAI 服务器平台。 目前我们的成本估算仅关注 T4 GPU，预计未来还会有 L4 GPU。 我们假设 Amazon EC2 G4 实例的保留定价为一年。

在这些条件下，在单个 T4 GPU 上处理 DICOM 工作负载的年度成本估计约为 7400 万美元，而与 CPU 管道相关的成本为 3.454 亿美元。 这意味着云支出显着减少，预测表明此类医院工作负载可能节省数亿美元。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/comparison-graphs-throughput-annual-cost-annual-energy-consumption-768x766.png)


在单个 T4 GPU 上，端到端吞吐量加速比 CPU 基准快大约 5 倍。 这种加速在新的 L4 GPU 上进一步增强，速度提高了约 12 倍。 当使用多个 GPU 实例时，性能几乎呈线性扩展。 例如，使用四个 T4 GPU 时，加速比可达到约 19 倍，而使用四个 L4 GPU 时，速度可提升至约 48 倍。

考虑到环境影响，能源效率是处理大量工作负载的数据中心的关键因素。 我们的计算表明，使用相应的基于 GPU 的硬件时，每年的能耗（以 GWh 为单位）会大大降低。 具体来说，单个L4系统的能耗约为CPU服务器的十二分之一。

对于类似于示例 DICOM 视频场景的工作负载（每分钟 500 小时的视频），每年的节能估计约为数百 GWh。 这些节能不仅具有经济效益，而且对环境也具有重要意义。 温室气体排放量的相应减少量是巨大的，类似于避免每年行驶数万辆客车的排放，每辆客车每年行驶约 11,000 英里。

## 为什么选择 nvImageCodec？
NVIDIA/nvImageCodec 库为开发人员提供了强大且高效的图像解码任务解决方案。 nvImageCodec 利用 NVIDIA GPU 的强大功能，提供加速的解码性能，非常适合需要高吞吐量和低延迟的应用程序。

### 主要特性
* GPU 加速：nvImageCodec 的突出特点之一是其 GPU 加速功能。 通过利用 NVIDIA GPU 的计算能力，nvImageCodec 显着加快了图像解码过程，从而可以更快地处理大型数据集。
* 无缝集成：nvImageCodec 与 Python 无缝集成，为开发人员的图像处理工作流程提供熟悉的环境。 借助用户友好的 API，将 nvImageCodec 集成到现有的 Python 项目中非常简单。
* 高性能：凭借优化的算法和并行处理，即使在处理复杂的图像解码任务时，nvImageCodec 也能提供卓越的性能。 无论您是解码 JPEG、JPEG 2000、TIFF 还是其他图像格式，nvImageCodec 都能确保快速高效的处理。
* 多功能性：从医学成像到计算机视觉应用，nvImageCodec 支持广泛的用例。 无论您处理的是灰度图像还是彩色图像，nvImageCodec 都能提供多功能性和灵活性来满足您的图像解码需求。
### 用例
* 医学成像：在医学成像领域，快速、准确的图像解码对于及时诊断和治疗至关重要。 借助 nvImageCodec，医疗保健专业人员可以快速、精确地解码医学图像，从而加快决策速度并改善患者治疗效果。
* 计算机视觉：在计算机视觉应用中，图像解码速度在对象检测和图像分类等实时处理任务中起着至关重要的作用。 通过利用 nvImageCodec 的 GPU 加速，开发人员可以实现高性能图像解码，从而增强应用程序的响应能力。
* 遥感：在遥感应用中，快速有效地解码大型卫星图像对于环境监测和灾害管理等各种任务至关重要。 借助 nvImageCodec，研究人员和分析人员可以轻松解码卫星图像，从而实现及时分析和决策。

### 如何获取nvImageCodec
获取 nvImageCodec 很简单。 您可以从多个来源下载它，例如 PyPI、NVIDIA 开发者专区，或直接从 GitHub 存储库下载。 下载后，您可以开始尝试编码和解码示例，以提高图像编解码器管道的效率。


![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/nvimagecodec-download-sources-screenshots.png)


![](![alt text](image.png))


## 如何批量解码高通量 JPEG 2000 医学图像
下面是一个 Python 示例，演示使用 nvImageCodec 库进行批量图像解码。 此示例说明如何使用 nvImageCodec 批量解码 HTJ2K 图像。 指定文件夹内的所有图像均以无损 HTJ2K 格式压缩，精度为 uint16 位。 输出确认所有医学图像均已成功解码，且质量没有任何损失（下图）。

```python
import os; import os.path
from matplotlib import pyplot as plt
from nvidia import nvimgcodec
 
 
dir = "htj2k_lossless"
image_paths = [os.path.join(dir, filename) for filename in os.listdir(dir)]
decode_params = nvimgcodec.DecodeParams(allow_any_depth = True, color_spec=nvimgcodec.ColorSpec.UNCHANGED)
nv_imgs = nvimgcodec.Decoder().read(image_paths, decode_params)
 
 
cols= 4
rows = (len(nv_imgs)+cols-1)//cols
fig, axes = plt.subplots(rows, cols); fig.set_figheight(2*rows); fig.set_figwidth(10)
for i in range(len(nv_imgs)):
    axes[i//cols][i%cols].set_title("%ix%i : %s"%(nv_imgs[i].height, nv_imgs[i].width, nv_imgs[i].dtype));
    axes[i//cols][i%cols].set_axis_off()
    axes[i//cols][i%cols].imshow(nv_imgs[i].cpu(), cmap='gray')

```
![](https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/nvimagecodec-batch-decoded-image-examples.png)


## 如何批量解码多个 JPEG 2000 图块
下面是一个 Python 示例，展示了使用 nvImageCodec 库对大图像进行基于图块的图像解码。 这演示了使用 nvImageCodec 解码大尺寸 JPEG 2000 压缩图像的过程。 每个图块代表一个感兴趣区域 (ROI)，尺寸为 512 x 512 像素。

解码过程包括将图像分割成图块，确定区域总数，然后使用 nvImageCodec 根据各个图块的索引对其进行解码，提供特定的图块解码信息。 生成的输出显示与不同图块相关的信息。

```python
from matplotlib import pyplot as plt
import numpy as np
import random; random.seed(654321)
from nvidia import nvimgcodec
jp2_stream = nvimgcodec.CodeStream('./B_37_FB3-SL_570-ST_NISL-SE_1708_lossless.jp2')
def get_region_grid(stream, roi_height, roi_width):
    regions = []
    num_regions_y = int(np.ceil(stream.height / roi_height))
    num_regions_x = int(np.ceil(stream.width / roi_width))
    for tile_y in range(num_regions_y):
        for tile_x in range(num_regions_x):
            tile_start = (tile_y * roi_height, tile_x * roi_width)
            tile_end = (np.clip((tile_y + 1) * roi_height, 0, stream.height), np.clip((tile_x + 1) * roi_width, 0, stream.width))
            regions.append(nvimgcodec.Region(start=tile_start, end=tile_end))
    print(f"{len(regions)} {roi_height}x{roi_width} regions in total")
    return regions
regions_native_tiles = get_region_grid(jp2_stream, jp2_stream.tile_height, jp2_stream.tile_width) # 512x512 tiles
 
 
dec_srcs = [nvimgcodec.DecodeSource(jp2_stream, region=regions_native_tiles[random.randint(0, len(regions_native_tiles)-1)]) for k in range(16)]
imgs = nvimgcodec.Decoder().decode(dec_srcs)
 
 
fig, axes = plt.subplots(4, 4)
fig.set_figheight(15)
fig.set_figwidth(15)
i = 0
for ax0 in axes:
    for ax1 in ax0:
        ax1.imshow(np.array(imgs[i].cpu()))
        i = i + 1

```



























































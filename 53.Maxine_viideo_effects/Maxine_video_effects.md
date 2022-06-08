# NVIDIA Maxine Video Effects SDK API Architecture


本节提供有关 Video Effects API 架构的信息。

# 1. 使用视频效果过滤器

要使用视频效果滤镜，需要先创建滤镜，设置滤镜的各种属性，然后加载、运行、销毁滤镜。

## 1.1 创建视频效果过滤器
以下是有关如何创建视频效果过滤器的信息。

调用 [NvVFX_CreateEffect()](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-createeffect) 函数，指定以下信息作为参数：

* NvVFX_EffectSelector 类型 有关详细信息，请参阅 [NvVFX_EffectSelector](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-effectselector)。
* 存储新创建的视频效果滤镜句柄的位置。


NvVFX_CreateEffect() 函数创建视频效果过滤器实例的句柄，以便在进一步的 API 调用中使用。

本示例创建一个 AI 绿幕视频效果滤镜。
```C++
NvCV_Status vfxErr = NvVFX_CreateEffect(NVVFX_FX_GREEN_SCREEN, &effectHandle);
```

## 1.2 设置模型文件夹的路径
视频效果过滤器需要一个神经网络模型来转换输入的静止图像或视频图像。 您必须将路径设置为包含描述过滤器要使用的模型的文件的文件夹。

调用 [`NvVFX_SetString()`](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-setstring) 函数，指定以下信息作为参数：
* 如[创建视频效果过滤器](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)中所述创建的过滤器句柄。
* 选择器字符串 NVVFX_MODEL_DIRECTORY。
* 一个以 null 结尾的字符串，指示模型文件夹的路径。

此示例将模型文件夹的路径设置为 `C:\Users\vfxuser\Documents\vfx\models`。


## 1.3 设置 CUDA 流
视频效果过滤器需要在其中运行的 CUDA 流。 有关 CUDA 流的信息，请参阅 [NVIDIA CUDA 工具包文档](https://docs.nvidia.com/cuda/)。

1. 通过调用以下函数之一初始化 CUDA 流。
    * CUDA 运行时 API 函数 `cudaStreamCreate()`。
    * 使用 [NvVFX_CudaStreamCreate()](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-cudastreamcreate) 函数来避免与 NVIDIA CUDA Toolkit 库链接。
2. 调用 [NvVFX_SetCudaStream() ](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-setcudastream)函数，提供以下信息作为参数：
    * 如创建视频效果过滤器中所述创建的过滤器句柄。
    * 选择器字符串 NVVFX_CUDA_STREAM。
    * 您在上一步中创建的 CUDA 流。
此示例设置通过调用 `NvVFX_CudaStreamCreate() `函数创建的 CUDA 流。

```C++
CUstream stream;
...
vfxErr = NvVFX_CudaStreamCreate (&stream);
...
vfxErr = NvVFX_SetCudaStream(effectHandle, NVVFX_CUDA_STREAM, stream);
```

## 1.4 创建和设置状态变量
网络摄像头去噪使用状态变量来跟踪输入视频流以去除噪声。 目前，没有其他视频效果过滤器使用状态变量。

SDK 用户负责完成以下任务：
* 创建状态变量。
  
    1.通过使用 NVVFX_STATE_SIZE 选择器字符串调用 NvVFX_GetU32() 来查询状态变量的大小。
    ```C++
    unsigned int stateSizeInBytes;
    vfxErr =  NvVFX_GetU32(effectHandle, NVVFX_STATE_SIZE, &stateSizeInBytes);
    ```
    2.使用 cudaMalloc() 为 GPU 中的状态变量分配必要的空间。
    ```C++
    void* state[1];
    cudaMalloc(&state[0], stateSizeInBytes);
    ```
    3.使用 cudaMemset() 将状态变量初始化为 0。
    ```C++
    cudaMemset(state[0], 0, stateSizeInByte);
    ```

* 将状态变量传递给 SDK。

    要将状态变量传递给 SDK，请使用 [NvVFX_SetObject()](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-setobject) 和 NVVFX_STATE 选择器字符串。
    ```C++
    vfxErr = NvVFX_SetObject(effectHandle, NVVFX_STATE, (void*)state);
    ```

* 释放状态变量内存。

    在状态变量被初始化和设置后，过滤器可以在图像或视频上运行。 在状态变量完成原始输入的处理后，您可以将该变量重新用于另一个图像/视频。

但是，在您可以在新输入上使用它之前，请使用 `cudaMemset()` 将状态变量重置为 0。 当状态变量不再使用时，使用 `cudaFree()` 释放为状态变量分配的内存。

```C++
cudaFree(state[0]);
```

## 1.5 设置输入和输出图像缓冲区
每个过滤器都将 GPU `NvCVImage` 结构作为输入，并在 GPU `NvCVImage` 结构中生成结果。有关 `NvCVImage` 的更多信息，请参阅 [NvCVImage](https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html) API 指南。这些图像是过滤器接受的 GPU 缓冲区。该应用程序通过将输入和输出缓冲区设置为所需参数来为过滤器提供输入和输出缓冲区。

视频效果过滤器需要在 GPU 缓冲区中提供输入。如果原始缓冲区是 CPU/GPU 类型或平面格式，则必须按照在 CPU 和 GPU 缓冲区之间传输图像中的说明进行转换。

以下是当前使用的格式列表：
* AI green screen: BGRu8 chunky --> Au8
* Background Blur: BGRu8 chunky + Au8 chunky --> BGRu8 chunky
* Upscale: RGBAu8 chunky --> RGBAu8 chunky
* ArtifactReduction: BGRf32 planar normalized --> BGRf32 planar normalized
* SuperRes: BGRf32 planar normalized --> BGRf32 planar normalized
* Transfer: anything --> anything
* Denoise: BGRf32 planar normalized --> BGRf32 planar normalized

**注意：BGRu8 chunky 是指一个 24 位像素，每个 B、G 和 R 像素分量都是 8 位。类似地，RGBAu8 chunky 指的是一个 32 位像素，每个 B、G、R 和 A 像素分量都是 8 位。**

**相比之下，BGRf32 平面是指每个像素分量的浮点精度，例如，B、G 和 R 像素分量各占 32 位。但是，由于这些是平面图像，它们不是紧凑的 96 位像素，并且存在三个 32 位平面，其中特定像素的每个组件可能以兆字节分隔。**


对于每个图像缓冲区，调用 [NvVFX_SetImage()](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-setimage) 函数，并指定以下信息作为参数：

* 如[创建视频效果过滤器](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)中所述创建的过滤器句柄。
* 表示您正在创建的缓冲区类型的选择器字符串：
* 对于输入图像缓冲区，使用 NVVFX_INPUT_IMAGE。
* 对于输出（掩码）图像缓冲区，使用 NVVFX_OUTPUT_IMAGE。
* 为输入或输出图像创建的 NvCVImage 对象的地址。
* 对于背景模糊，使用 NVVFX_INPUT_IMAGE_1 传递作为分割掩码的第二个输入。


此示例创建一个输入图像缓冲区。

```C++
NvCVImage srcGpuImage;
...
vfxErr = NvCVImage_Alloc(&srcGpuImage, 960, 540, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1)
...
vfxErr = NvVFX_SetImage(effectHandle, NVVFX_INPUT_IMAGE, &srcGpuImage);
```
此示例创建一个输出图像缓冲区。
```C++
NvCVImage srcGpuImage;
...
vfxErr = NvCVImage_Alloc(&dstGpuImage, 960, 540, NVCV_A, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1))
...
vfxErr = NvVFX_SetImage(effectHandle, NVVFX_OUTPUT_IMAGE, &dstGpuImage);
```


## 1.6 设置和获取视频效果滤镜的其他参数
在加载和运行视频效果过滤器之前，请设置过滤器所需的任何其他参数。 Video Effects SDK 为此提供了类型安全的集访问器函数。 如果您需要具有 set 访问器函数的参数的值，请使用相应的 get 访问器函数。

以下是当前使用的格式列表：
* 如[创建视频效果过滤器](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)中所述创建的过滤器句柄。
* 您要访问的参数的选择器字符串。
* 您要设置的值或指向存储您要获取的值的位置的指针。


此示例将 AI 绿屏过滤器的模式设置为最快性能。

```C++
vfxErr = NvVFX_SetU32(effectHandle, NVVFX_MODE, 1);
```

### 1.6.1 示例：设置 AI 绿屏的过滤模式
这是 AI 绿屏过滤器的任务示例。

AI绿屏滤镜支持以下操作模式：
* 质量模式，提供最高质量的结果（默认）。
* 性能模式，提供最快的性能。
调用 `NvVFX_SetU32()` 函数，指定以下信息作为参数：
* 如[创建视频效果过滤器](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)中所述创建的过滤器句柄。
* 选择器字符串 NVVFX_MODE。 
* 一个整数，表示您想要的操作模式：
    * 0：质量模式
    * 1：性能模式
此示例将模式设置为 Performance。
```C++
vfxErr = NvVFX_SetU32 (effectHandle, NVVFX_MODE, 1);
```


## 1.6.2 示例：为 AI 绿屏启用 CUDA 图形优化
AI绿屏滤镜支持CUDA Graph Optimization，可以通过设置正确的`NVVFX_CUDA_GRAPH`值来启用。

调用 [`NvVFX_SetU32()`](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#nvvfx-setu32) 函数并指定以下信息作为参数：
* 已创建的滤镜句柄（有关详细信息，请参阅[创建视频效果滤镜](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)）。
* NVVFX_CUDA_GRAPH 选择器字符串。
* 以下整数，表示优化状态：
    * 0：关闭
    * 1：开
**注意：对于 AI 绿屏过滤器，CUDA 图优化的默认状态是关闭。**

在调用NvVFX_Load()方法之前调用如下set方法初始化AI绿屏效果：
```C++
vfxErr = NvVFX_SetU32 (effectHandle, NVVFX_CUDA_GRAPH, 1);
```

* 如果启用了 CUDA 图优化，则第一次调用 NVVFX_RUN() 会初始化优化。
  
  在图初始化期间，如果发生错误，则返回 `NVCV_ERR_CUDA` 错误代码。

* 如果在初始化后调用 set 方法，则 `NvVFX_Run()` 调用将返回 `NVCV_ERR_INITIALIZATION` 错误代码。


启用 CUDA 图优化可减少内核启动开销，并可能提高整体性能。为了验证改进，我们建议开发人员通过在 Off 和 On 之间切换状态来使用 CUDA Graph Optimization 测试他们的应用程序。


## 1.7 NVIDIA Video Effects SDK 访问器函数汇总
下表提供了有关 SDK 访问器函数的详细信息。

<div class="tablenoborder"><a name="summ-sdk-accessor-functions-vfx__table_btz_hzl_xpb" shape="rect">
                                    <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="summ-sdk-accessor-functions-vfx__table_btz_hzl_xpb" class="table" frame="border" border="1" rules="all">
                                    <caption><span class="tablecap">Table 2. Video Effects SDK Accessor Functions</span></caption>
                                    <thead class="thead" align="left">
                                       <tr class="row">
                                          <th class="entry" align="left" valign="top" id="d54e1345" rowspan="1" colspan="1">Property Type</th>
                                          <th class="entry" align="left" valign="top" id="d54e1348" rowspan="1" colspan="1">Data Type</th>
                                          <th class="entry" align="left" valign="top" id="d54e1351" rowspan="1" colspan="1">Set and Get Accessor Function </th>
                                       </tr>
                                    </thead>
                                    <tbody class="tbody">
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">32-bit unsigned integer</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">unsigned int</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetU32()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetU32()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">32-bit signed integer</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">int</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetS32()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetS32()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">Single-precision (32-bit) floating-point number</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">float</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetF32()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetF32()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">Double-precision (64-bit) floating point number </td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">double</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetF64()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetF64()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">64-bit unsigned integer</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">unsigned long long</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetU64()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetU64()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">Image buffer</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">NvCVImage</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetImage()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetImage() </samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">Object </td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">void</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetObject()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetObject()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">Character string</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">const char*</td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetString()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetString()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1345" colspan="1">CUDA stream</td>
                                          <td class="entry" rowspan="2" align="left" valign="top" headers="d54e1348" colspan="1">CUstream </td>
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_SetCudaStream()</samp></td>
                                       </tr>
                                       <tr class="row">
                                          <td class="entry" align="left" valign="top" headers="d54e1351" rowspan="1" colspan="1"><samp class="ph codeph">NvVFX_GetCudaStream()</samp></td>
                                       </tr>
                                    </tbody>
                                 </table>
                              </div>


## 1.8 获取有关过滤器及其参数的信息
以下是有关如何获取有关过滤器及其参数的信息的一些信息。

要获取有关过滤器及其参数的信息，请调用 `NvVFX_GetString()` 函数，指定 `NvVFX_ParameterSelector` 类型定义的 `NVVFX_INFO` 类型。
```C++
NvCV_Status NvVFX_GetString(
  NvVFX_Handle obj,
  NVVFX_INFO,
  const char **str
);
```

## 1.9 获取所有可用效果的列表
以下是有关如何获取可用效果列表的信息。

要获取可用效果的列表，请调用 `NvVFX_GetString()` 函数，为 `NvVFX_Handle` 对象句柄指定 `NULL`。
```C++
NvCV_Status NvVFX_GetString(NULL, NVVFX_INFO, const char **str);
```


# 2. 加载视频效果滤镜
加载过滤器选择并加载效果模型并验证为过滤器设置的参数。

**注意：某些视频效果滤镜的设置只能在加载滤镜后进行修改。**

要加载视频效果滤镜，请调用 `NvVFX_Load()` 函数，指定创建的滤镜句柄，如[创建视频效果滤镜](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)中所述。
```C++
vfxErr = NvVFX_Load(effectHandle);
```

**注意：如果使用 set 访问器函数更改过滤器参数，为了使更改生效，过滤器可能需要在运行之前重新加载。**


# 3. 运行视频效果过滤器
加载视频效果过滤器后，运行过滤器以应用所需的效果。 运行过滤器时，会读取输入 GPU 缓冲区的内容，应用视频效果过滤器，并将输出写入输出 GPU 缓冲区。

要运行视频效果滤镜，请调用 `NvVFX_Run()` 函数。 在调用 `NvVFX_Run()` 函数时，将以下信息作为参数传递：
* 如[创建视频效果过滤器](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html#create-vfx-filter)中所述创建的过滤器句柄。
* 一个整数值，用于指定过滤器是异步运行还是同步运行：
    * 1：过滤器是异步运行的。
    * 0：过滤器同步运行。

此示例异步运行视频效果滤镜并调用` NvCVImage_Transfer()` 函数将输出复制到 CPU 缓冲区中。

```C++
vfxErr = NvVFX_Run(effectHandle, 1);
vfxErr = NvCVImage_Transfer()
```

# 4. 销毁视频效果过滤器
当不再需要视频效果过滤器时，将其销毁以释放分配给过滤器的资源和内存。

要销毁视频效果滤镜，请调用 `NvVFX_DestroyEffect()` 函数，指定按照创建视频效果滤镜中所述创建的滤镜句柄。
```C++
NvVFX_DestroyEffect(effectHandle);
```

您可以使用 `Video Effects SDK` 使应用程序能够将效果过滤器应用于视频。 Video Effects API 是面向对象的，但除了 C++ 之外，C 也可以访问它。

# 5.在 GPU 或 CPU 缓冲区上处理图像帧
效果过滤器接受图像缓冲区作为 NvCVImage 对象。图像缓冲区可以是 CPU 或 GPU 缓冲区，但出于性能原因，效果过滤器需要 GPU 缓冲区。 Video Effects SDK 提供了将图像表示转换为 NvCVImage 以及在 CPU 和 GPU 缓冲区之间传输图像的功能。

有关 NvCVImage 的更多信息，请参阅 [NvCVImage API 指南](https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html)。本部分简要介绍了 Video Effects SDK 最常用的功能。

## 5.1 将图像表示转换为 NvCVImage 对象
Video Effects SDK 提供了将 OpenCV 图像和其他图像表示转换为 NvCVImage 对象的功能。每个函数都在现有缓冲区周围放置一个包装器。包装器防止在调用包装器的析构函数时释放缓冲区。

### 5.1.1。将 OpenCV 图像转换为 NvCVImage 对象
这是有关如何将 OpenCV 图像转换为 NvCVImage 对象的信息。

**注意：使用 NVIDIA Video Effects SDK 专门为 RGB OpenCV 图像提供的包装函数。**

* 要为 OpenCV 图像创建 NvCVImage 对象包装器，请使用 NVWrapperForCVMat() 函数。
    ```C++
    //Allocate source and destination OpenCV images
    cv::Mat srcCVImg(   );
    cv::Mat dstCVImg(...);
 
    // Declare source and destination NvCVImage objects
    NvCVImage srcCPUImg;
    NvCVImage dstCPUImg;
 
    NVWrapperForCVMat(&srcCVImg, &srcCPUImg);
    NVWrapperForCVMat(&dstCVImg, &dstCPUImg);
    ```
* 要为 NvCVImage 对象创建 OpenCV 图像包装器，请使用 CVWrapperForNvCVImage() 函数。
    ```C++
    // Allocate source and destination NvCVImage objects
    NvCVImage srcCPUImg(...);
    NvCVImage dstCPUImg(...);
 
    //Declare source and destination OpenCV images
    cv::Mat srcCVImg;
    cv::Mat dstCVImg;
 
    CVWrapperForNvCVImage (&srcCPUImg, &srcCVImg);
    CVWrapperForNvCVImage (&dstCPUImg, &dstCVImg);

    ```

### 5.1.2 将 GPU 或 CPU 缓冲区上的图像帧转换为 NvCVImage 对象
以下是有关如何将 CPU 或 GPU 缓冲区上的图像帧转换为 NvCVImage 对象的信息。

调用 `NvCVImage_Init()` 函数在现有缓冲区 (`srcPixelBuffer`) 周围放置一个包装器。

```C++
NvCVImage src_gpu;
vfxErr = NvCVImage_Init(&src_gpu, 640, 480, 1920, srcPixelBuffer, NVCV_BGR, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU);
 
NvCVImage src_cpu;
vfxErr = NvCVImage_Init(&src_cpu, 640, 480, 1920, srcPixelBuffer, NVCV_BGR, NVCV_U8, NVCV_INTERLEAVED, NVCV_CPU);
```

### 5.1.3 将解码帧从 NvDecoder 转换为 NvCVImage 对象
这是有关将解码帧从 NvDecoder 转换为 NvCVImage 对象的信息。

调用 `NvCVImage_Transfer()` 函数将 NvDecoder 提供的解码帧从解码的像素格式转换为 Video Effects SDK 功能所需的格式。 以下示例显示了从 NV12 转换为 BGRA 像素格式的解码帧。

```C++
vCVImage decoded_frame, BGRA_frame, stagingBuffer;
NvDecoder dec;
 
//Initialize decoder...
//Assuming dec.GetOutputFormat() == cudaVideoSurfaceFormat_NV12
 
//Initialize memory for decoded frame
NvCVImage_Init(&decoded_frame, dec.GetWidth(), dec.GetHeight(), dec.GetDeviceFramePitch(), NULL, NVCV_YUV420, NVCV_U8, NVCV_NV12, NVCV_GPU, 1);
decoded_frame.colorSpace = NVCV_709 | NVCV_VIDEO_RANGE | NVCV_CHROMA_ COSITED;
 
//Allocate memory for BGRA frame
NvCVImage_Alloc(&BGRA_frame, dec.GetWidth(), dec.GetHeight(), NVCV_BGRA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);
 
decoded_frame.pixels = (void*)dec.GetFrame();
 
//Convert from decoded frame format(NV12) to desired format(BGRA)
NvCVImage_Transfer(&decoded_frame, &BGRA_frame, 1.f, stream, & stagingBuffer);
```
**注意：上面的示例假定了高清内容的典型色彩空间规范。 SD 通常使用 NVCV_601。 有 8 种可能的组合，您应该使用与视频标题中描述的视频相匹配的组合，或者通过反复试验继续进行。**

以下是一些附加信息：
* 如果颜色不正确，请交换 709<->601。
* 如果它们被冲掉或销毁，请交换 VIDEO<->FULL。
* 如果颜色水平移动，则交换 INTSTITIAL<->COSITED。

### 5.1.4 将 NvCVImage 对象转换为可以被 NvEncoder 编码的 Buffer
以下是有关如何使用 NvEncoder 将 NvCVImage 对象转换为缓冲区的信息。

要通过 `NvEncoder` 将 `NvCVImage` 转换为在编码期间使用的像素格式，如果需要，请调用 `NvCVImage_Transfer()` 函数。 以下示例显示了以 `BGRA `像素格式编码的帧。
```C++
//BGRA frame is 4-channel, u8 buffer residing on the GPU
NvCVImage BGRA_frame;
NvCVImage_Alloc(&BGRA_frame, dec.GetWidth(), dec.GetHeight(), NVCV_BGRA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);
 
//Initialize encoder with a BGRA output pixel format
using NvEncCudaPtr = std::unique_ptr<NvEncoderCuda, std::function<void(NvEncoderCuda*)>>;
NvEncCudaPtr pEnc(new NvEncoderCuda(cuContext, dec.GetWidth(), dec.GetHeight(), NV_ENC_BUFFER_FORMAT_ARGB));
pEnc->CreateEncoder(&initializeParams);
//...
 
std::vector<std::vector<uint8_t>> vPacket;
//Get the address of the next input frame from the encoder
const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();
 
//Copy the pixel data from BGRA_frame into the input frame address obtained above
NvEncoderCuda::CopyToDeviceFrame(cuContext,
                        BGRA_frame.pixels,
                        BGRA_frame.pitch,
                        (CUdeviceptr)encoderInputFrame->inputPtr,
                        encoderInputFrame->pitch,
                        pEnc->GetEncodeWidth(),
                        pEnc->GetEncodeHeight(),
                        CU_MEMORYTYPE_DEVICE,
                        encoderInputFrame->bufferFormat,
                        encoderInputFrame->chromaOffsets,
                        encoderInputFrame->numChromaPlanes);
pEnc->EncodeFrame(vPacket);
```

## 5.2.分配 NvCVImage 对象缓冲区
您可以使用 `NvCVImage` 分配构造函数或图像函数为 `NvCVImage `对象分配缓冲区。在这两个选项中，当图像超出范围时，析构函数会自动释放缓冲区。

### 5.2.1。使用 NvCVImage 分配构造函数分配缓冲区
`NvCVImage` 分配构造函数创建一个已分配内存并已初始化的对象。有关详细信息，请参阅[分配构造函数](https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html#allocation-constructor)。
分配构造函数的最后三个可选参数决定了生成的 `NvCVImage `对象的属性：
* 像素组织决定了蓝色、绿色和红色是在不同的平面中还是交错的。
* 内存类型决定了缓冲区是驻留在 GPU 上还是 CPU 上。
* 字节对齐决定了连续扫描线之间的间隙。

以下示例展示了如何使用分配构造函数的最后三个可选参数来确定 NvCVImage 对象的属性。
* 此示例创建一个对象，而不设置分配构造函数的最后三个可选参数。在这个对象中，蓝色、绿色和红色分量交错在每个像素中，缓冲区驻留在 CPU 上，字节对齐是默认对齐。

```C++
NvCVImage cpuSrc(
  srcWidth,
  srcHeight,
  NVCV_BGR,
  NVCV_U8
);
```

* 此示例通过显式设置最后三个可选参数来创建与上一个示例具有相同像素组织、内存类型和字节对齐的对象。 与前面的示例一样，蓝色、绿色和红色分量在每个像素中交错，缓冲区驻留在 CPU 上，并且字节对齐是默认设置，即针对最大性能进行了优化。
```C++
NvCVImage src(
  srcWidth,
  srcHeight,
  NVCV_BGR,
  NVCV_U8,
  NVCV_INTERLEAVED,
  NVCV_CPU,
  0
);
```

* 这个例子创建了一个对象，其中蓝色、绿色和红色分量位于不同的平面中，缓冲区位于 GPU 上，字节对齐确保一个扫描线和下一个扫描线之间不存在间隙。

```C++
NvCVImage gpuSrc(
  srcWidth,
  srcHeight,
  NVCV_BGR,
  NVCV_U8,
  NVCV_PLANAR,
  NVCV_GPU,
  1
);
```

### 5.2.2 使用图像函数分配缓冲区
通过声明一个空图像，您可以推迟缓冲区分配。

1. 声明一个空的 NvCVImage 对象。
```C++
NvCVImage xfr;
```

2. 为图像分配或重新分配缓冲区。
    * 要分配缓冲区，请调用 `NvCVImage_Alloc()` 函数。
        
        当图像是状态结构的一部分时，以这种方式分配缓冲区，直到稍后您才知道图像的大小。

    * 要重新分配缓冲区，请调用 `NvCVImage_Realloc()`。
        
        该函数在释放缓冲区并调用 `NvCVImage_Alloc()` 之前检查分配的缓冲区并在缓冲区足够大时对其进行整形。


## 5.3 在 CPU 和 GPU 缓冲区之间传输图像
如果输入和输出图像缓冲区的内存类型不同，应用程序可以在 CPU 和 GPU 缓冲区之间传输图像。

### 5.3.1 将输入图像从 CPU 缓冲区传输到 GPU 缓冲区
以下是有关如何将输入图像从 CPU 缓冲区传输到 GPU 缓冲区的信息。

要将图像从 CPU 传输到 GPU 缓冲区并进行转换，请给出以下代码：
```C++
NvCVImage srcCpuImg(width, height, NVCV_RGB, NVCV_U8, NVCV_INTERLEAVED,
                    NVCV_CPU, 1);
NvCVImage dstGpuImg(width, height, NVCV_BGR, NVCV_F32, NVCV_PLANAR,
                    NVCV_GPU, 1);
```

1. 通过以下方式之一创建一个 NvCVImage 对象以用作暂存 GPU 缓冲区：
    * 为避免在视频管道中分配内存，请在初始化阶段创建一个 GPU 缓冲区，其尺寸和格式与 CPU 图像相同。
    ```C++
    NvCVImage stageImg(srcCpuImg.width, srcCpuImg.height,
          srcCpuImg.pixelFormat, srcCpuImg.componentType,
          srcCpuImg.planar, NVCV_GPU);
    ```
    * 为了简化您的应用程序代码，您可以在初始化阶段声明一个空的暂存缓冲区。
    ```C++
    NvCVImage stageImg;
    ```
    如果需要，将根据需要分配或重新分配适当大小的缓冲区。

2. 调用 NvCVImage_Transfer() 函数将源 CPU 缓冲区内容通过暂存 GPU 缓冲区复制到最终 GPU 缓冲区中。

    ```C++
    // Transfer the image from the CPU to the GPU, perhaps with conversion.
    NvCVImage_Transfer(&srcCpuImg, &dstGpuImg, 1.0f, stream, &stageImg);
    ```
    无论图像大小如何，相同的暂存缓冲区都可以在不同上下文中的多个 `NvCVImage_Transfer` 调用中重复使用，并且如果它是持久的，则可以避免缓冲区分配。

### 5.3.2 将输出图像从 GPU 缓冲区传输到 CPU 缓冲区
以下是有关如何将输出图像从 GPU 缓冲区传输到 CPU 缓冲区的信息。

要将图像从 GPU 传输到 CPU 缓冲区并进行转换，请给出以下代码：
```C++
To transfer an image from the GPU to a CPU buffer with conversion, given the following code:
NvCVImage srcGpuImg(width, height, NVCV_BGR, NVCV_F32, NVCV_PLANAR,
                    NVCV_GPU, 1);
NvCVImage dstCpuImg(width, height, NVCV_BGR, NVCV_U8, NVCV_INTERLEAVED,
                    NVCV_CPU, 1);
```

1. 通过以下方式之一创建一个 NvCVImage 对象以用作暂存 GPU 缓冲区：
    * 为避免在视频管道中分配内存，请在初始化阶段创建一个与 CPU 图像具有相同尺寸和格式的 GPU 缓冲区。
    ```C++
    NvCVImage stageImg(dstCpuImg.width, dstCpuImg.height,
    dstCPUImg.pixelFormat, dstCPUImg.componentType,
    dstCPUImg.planar, NVCV_GPU);
    ```
    * 为了简化您的应用程序代码，您可以在初始化阶段声明一个空的暂存缓冲区。
    ```C++
    NvCVImage stageImg;
    ```

    如果需要，将根据需要分配或重新分配适当大小的缓冲区。

    有关 NvCVImage 的更多信息，请参阅 [NvCVImage API 指南](https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html)。

    如果 NvCVImage_Transfer 是持久的，则可以重复使用相同的暂存缓冲区，而无需在 `NvCVImage_Transfer` 中重新分配。   
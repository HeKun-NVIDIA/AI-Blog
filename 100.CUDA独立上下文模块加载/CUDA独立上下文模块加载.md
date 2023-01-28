# CUDA独立上下文模块加载
![](cuda-image-16x9-1-e1670969622882.jpg)

大多数 CUDA 开发人员都熟悉 cuModuleLoad API 及其对应项，用于将包含设备代码的模块加载到 CUDA 上下文中。 在大多数情况下，您希望在所有设备上加载相同的设备代码。 这需要将设备代码显式加载到每个 CUDA 上下文中。 此外，不控制上下文创建和销毁的库和框架必须跟踪它们以显式加载和卸载模块。

这篇文章讨论了 CUDA 12.0 中引入的上下文独立加载，它解决了这些问题。


## 上下文相关的加载(Context-dependent loading)
传统上，模块加载总是与 CUDA 上下文相关联。 下面的代码示例显示了将相同的设备代码加载到两个设备然后在它们上启动内核的传统方法。

```cpp
// Device 0
cuDeviceGet(&device0, 0);
cuDevicePrimaryCtxRetain(&ctx0, device0);
cuModuleLoad(&module0, “myModule.cubin”);
// Device 1
cuDeviceGet(&device1, 1);
cuDevicePrimaryCtxRetain(&ctx1, device1);
cuModuleLoad(&module1, “myModule.cubin”);
```

在每个设备上启动内核需要您检索每个模块的 CU 函数，如以下代码示例所示：
```cpp
// Device 0
cuModuleGetFuntion(&function0, module0, “myKernel”);
cuLaunchKernel(function0, …);
// Device 1
cuModuleGetFuntion(&function1, module1, “myKernel”);
cuLaunchKernel(function1, …);
```

这会增加应用程序中的代码复杂性，因为您必须检索和跟踪每个上下文和每个模块的类型。 您还必须使用 cuModuleUnload API 显式卸载每个模块。

当库或框架主要使用 CUDA 驱动程序 API 来加载它们自己的模块时，问题会加剧。 他们可能无法完全控制应用程序拥有的上下文的生命周期。
```cpp
// Application code

libraryInitialize();
cuDevicePrimaryCtxRetain(&ctx0, device0);
libraryFunc();
cuDevicePrimaryCtxRetain(&ctx0, device1);
libraryFunc();
libraryDeinitialize();

// Library code

libraryInitialize() {
  map<CUcontext, CUmodule> moduleContextMap;
}

libraryFunc() {
  cuCtxGetCurrent(&ctx);
  if (!moduleContextMap.contains(ctx)){
    cuModuleLoad(&module, “myModule.cubin”);
    moduleContextMap[ctx] = module;
  }
  else {
    module = moduleContextMap[ctx];
  }

  cuModuleGetFuntion(&function, module, “myKernel”);
  cuLaunchKernel(function, …);
}

libraryDeinitialize() {
  moduleContextMap.clear();
}
```

在代码示例中，库必须检查新上下文并将模块显式加载到其中。 它还必须维护状态以检查模块是否已加载到上下文中。

理想情况下，状态可以在上下文被销毁后释放。 但是，如果库无法控制上下文的生命周期，这是不可能的。

这意味着资源的释放必须延迟到库取消初始化。 这不仅增加了代码的复杂性，而且还导致库占用资源的时间超过了它必须占用的时间，可能会拒绝应用程序的另一部分使用该内存。

另一种选择是让库和框架对用户施加额外的限制，以确保他们对资源分配和清理有足够的控制权。


## 上下文独立的加载(Context-independent loading)
CUDA 12.0 通过添加 cuLibrary* 和 cuKernel* API 引入了上下文无关加载，从而解决了这些问题。 通过上下文独立加载，在应用程序创建和销毁上下文时，CUDA 驱动程序会自动将模块加载和卸载到每个 CUDA 上下文中。

```c++
// Load library
cuLibraryLoadFromFile(&library,“myModule.cubin”, …);
cuLibraryGetKernel(&kernel, library, “myKernel”);

// Launch kernel on the primary context of device 0
cuDevicePrimaryCtxRetain(&ctx0, device0);
cuLaunchKernel((CUkernel)kernel, …);

// Launch kernel on the primary context of device 1
cuDevicePrimaryCtxRetain(&ctx1, device1);
cuLaunchKernel((CUkernel)kernel, …);

// Unload library
cuLibraryUnload(library);
```
如代码示例所示，cuLibraryLoadFromFile API 负责在创建或初始化上下文时加载模块。 在示例中，这是在 cuDevicePrimaryCtxRetain 期间完成的。

此外，您现在可以使用与上下文无关的句柄 CUkernel 启动内核，而不必维护每个上下文的 CUfunction。 cuLibraryGetKernel 检索设备函数 myKernel 的上下文无关句柄。 然后可以通过指定与上下文无关的句柄 CUkernel 来使用 cuLaunchKernel 启动设备功能。 CUDA 驱动程序负责根据此时处于活动状态的上下文在适当的上下文中启动设备功能。

库和框架现在可以分别在初始化和取消初始化期间简单地加载和卸载模块一次。
```cpp
// Application code

libraryInitialize();
cuDevicePrimaryCtxRetain(&ctx0, device0);
libraryFunc();
cuDevicePrimaryCtxRetain(&ctx0, device1);
libraryFunc();
libraryDeinitialize();

// Library code

libraryInitialize() {
  cuLibraryLoadFromFile(&library,“myModule.cubin”, …);
  cuLibraryGetKernel(&kernel, library, “myKernel”);
}

libraryFunc() {
  cuLaunchKernel((CUkernel)kernel, …);
}

libraryDeinitialize() {
  cuLibraryUnload(library);
}
```
该库不再需要维护和跟踪每个上下文的状态。 context-independent loading的设计使得CUDA driver能够跟踪模块和context，进行加载和卸载模块的工作。

## 访问 `__managed__` 变量
托管变量可以从设备和主机代码中引用。 例如，可以查询托管变量的地址，或者可以直接从设备或主机函数读取或写入它。 与具有创建它的 CUDA 上下文的生命周期的 `__device__` 变量不同，属于模块的 `__managed__` 变量指向跨所有 CUDA 上下文甚至设备的相同内存。

在 CUDA 12.0 之前，无法通过驱动程序 API 检索到跨 CUDA 上下文唯一的托管变量的句柄。 CUDA 12.0 引入了一个新的驱动程序 API cuLibraryGetManaged，这使得跨 CUDA 上下文获取唯一句柄成为可能。

## 开始使用与上下文无关的加载
在本文中，我们介绍了新的 CUDA 驱动程序 API，它们提供了独立于 CUDA 上下文加载设备代码的能力。 我们还讨论了用于启动内核的与上下文无关的句柄。 与传统的加载机制相比，它们共同提供了一种在 GPU 上加载和执行代码的更简单方法，从而降低了代码复杂性并避免了维护每个上下文状态的需要。

要开始使用这些 API，请下载 [CUDA 驱动程序和工具包版本 12 或更高版本](https://developer.nvidia.com/cuda-toolkit)。 有关 cuLibrary* 和 cuKernel* API 的更多信息，请参阅 [CUDA Driver API ](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)文档。




























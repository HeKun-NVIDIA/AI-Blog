# 在CUDA编程模型中利用Tensor Core加速矩阵运算
C++ warp矩阵运算利用Tensor Cores来加速 `D=A*B+C` 形式的矩阵问题。 计算能力 7.0 或更高版本的设备的混合精度浮点数据支持这些操作。 这需要一个warp中所有线程的合作。 此外，仅当条件在整个 [warp](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) 中的计算结果相同时，才允许在条件代码中执行这些操作，否则代码执行可能会挂起。

##    1. Description
以下所有函数和类型都在命名空间 `nvcuda::wmma` 中定义。 Sub-byte操作被视为预览版，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。 这个额外的功能在 nvcuda::wmma::experimental 命名空间中定义。
```C++
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

`fragment`:

包含矩阵的一部分的重载类，分布在warp中的所有线程中。 矩阵元素到`fragment`内部存储的映射是未指定的，并且在未来的架构中可能会发生变化。

只允许模板参数的某些组合。 第一个模板参数指定片段将如何参与矩阵运算。 可接受的使用值是：
* `matrix_a` 当`fragment` 用作第一个被乘数时，A
* `matrix_b` 当`fragment`用作第二个被乘数时，B
* 当`fragment`用作源或目标累加器（分别为 C 或 D）时的累加器。

`m、n 和 k` 大小描述了参与乘法累加操作的warp-wide矩阵块的形状。 每个tile的尺寸取决于它的作用。 对于 `matrix_a`，图块的尺寸为 `m x k`； 对于 `matrix_b`，维度是 `k x n`，累加器块是 `m x n`。

对于被乘数，数据类型 `T` 可以是 `double、float、__half、__nv_bfloat16、char 或 unsigned char`，对于累加器，可以是 `double、float、int 或 __half`。 如[元素类型和矩阵大小](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-type-sizes)中所述，支持累加器和被乘数类型的有限组合。 必须为 `matrix_a` 和 `matrix_b` 片段指定 `Layout` 参数。 `row_major` 或 `col_major` 分别表示矩阵***行或列***中的元素在内存中是连续的。 累加器矩阵的 `Layout` 参数应保留默认值 `void`。 仅当按如下所述加载或存储累加器时才指定行或列布局。


`load_matrix_sync`:

等到所有warp通道(lanes)都到达 `load_matrix_sync`，然后从内存中加载矩阵片段 `a`。 `mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。 `ldm` 描述连续行（对于行主序）或列（对于列主序）之间的元素跨度，对于 `__half` 元素类型必须是 8 的倍数，对于浮点元素类型必须是 4 的倍数。 （即，两种情况下都是 16 字节的倍数）。 如果`fragment`是累加器，则布局参数必须指定为 `mem_row_major` 或 `mem_col_major`。 对于 `matrix_a` 和 `matrix_b` 片段，`Layout`是从`fragment`的`Layout`参数中推断出来的。 a 的 `mptr、ldm、layout` 和所有模板参数的值对于 warp 中的所有线程必须相同。 这个函数必须被warp中的所有线程调用，否则结果是未定义的。

`store_matrix_sync`:

等到所有warp通道都到达 `store_matrix_sync`，然后将矩阵片段 a 存储到内存中。 `mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。 `ldm` 描述连续行（对于行主序）或列（对于列主序）之间的元素跨度，对于` __half` 元素类型必须是 8 的倍数，对于浮点元素类型必须是 4 的倍数。 （即，两种情况下都是 16 字节的倍数）。 输出矩阵的布局必须指定为 `mem_row_major` 或 `mem_col_major`。 a 的 `mptr、ldm、layout` 和所有模板参数的值对于 warp 中的所有线程必须相同。

`fill_fragment`:

用常量 v 填充矩阵片段。由于未指定矩阵元素到每个片段的映射，因此该函数通常由 warp 中的所有线程调用，并具有共同的 v 值。

`mma_sync`:

等到所有`warp lanes`都到达`mma_sync`，然后执行warp同步的矩阵乘法累加操作`D=A*B+C`。 还支持原位(in-place)操作，`C=A*B+C`。 对于 warp 中的所有线程，每个矩阵片段的 `satf` 和模板参数的值必须相同。 此外，模板参数 `m、n 和 k` 必须在片段 `A、B、C 和 D` 之间匹配。该函数必须由 warp 中的所有线程调用，否则结果未定义。

如果 `satf`（饱和到有限值--saturate to finite value）模式为真，则以下附加数值属性适用于目标累加器：
* 如果元素结果为+Infinity，则相应的累加器将包含+MAX_NORM
* 如果元素结果为 -Infinity，则相应的累加器将包含 -MAX_NORM
* 如果元素结果为 NaN，则对应的累加器将包含 +0

由于未指定矩阵元素到每个线程片段的映射，因此必须在调用 `store_matrix_sync` 后从内存（共享或全局）访问单个矩阵元素。 在 warp 中的所有线程将对所有片段元素统一应用元素操作的特殊情况下，可以使用以下`fragment`类成员实现直接元素访问。

```C++
enum fragment<Use, m, n, k, T, Layout>::num_elements;
T fragment<Use, m, n, k, T, Layout>::x[num_elements];
```

例如，以下代码将累加器矩阵缩小一半。
```C++
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
float alpha = 0.5f; // Same value for all threads in warp
/*...*/
for(int t=0; t<frag.num_elements; t++)
frag.x[t] *= alpha; 
```

##    2. Alternate Floating Point
Tensor Core 支持在具有 8.0 及更高计算能力的设备上进行替代类型的浮点运算。

`__nv_bfloat16`:

此数据格式是另一种 `fp16 `格式，其范围与 `f32` 相同，但精度降低（7 位）。 您可以直接将此数据格式与 `cuda_bf16.h` 中提供的 `__nv_bfloat16` 类型一起使用。 具有 `__nv_bfloat16` 数据类型的矩阵片段需要与浮点类型的累加器组合。 支持的形状和操作与 `__half` 相同。

`tf32`:

这种数据格式是 `Tensor Cores` 支持的特殊浮点格式，范围与 f32 相同，但精度降低（>=10 位）。这种格式的内部布局是实现定义的。为了在 `WMMA` 操作中使用这种浮点格式，输入矩阵必须手动转换为 `tf32` 精度。

为了便于转换，提供了一个新的内联函数 `__float_to_tf32`。虽然内联函数的输入和输出参数是浮点类型，但输出将是 `tf32`。这个新精度仅适用于张量核心，如果与其他浮点类型操作混合使用，结果的精度和范围将是未定义的。

一旦输入矩阵（`matrix_a` 或 `matrix_b`）被转换为 `tf32` 精度，具有`precision::tf32` 精度的片段和`load_matrix_sync` 的`float` 数据类型的组合将利用此新功能。两个累加器片段都必须具有浮点数据类型。唯一支持的矩阵大小是 `16x16x8 (m-n-k)`。

片段的元素表示为浮点数，因此从 `element_type<T>` 到 `storage_element_type<T>` 的映射是：
```C++
precision::tf32 -> float
```

##    3. Double Precision
`Tensor Core` 支持计算能力 8.0 及更高版本的设备上的双精度浮点运算。 要使用这个新功能，必须使用具有 `double` 类型的片段。 `mma_sync` 操作将使用 `.rn`（四舍五入到最接近的偶数）舍入修饰符执行。

##    4. Sub-byte Operations
 
Sub-byte `WMMA` 操作提供了一种访问 Tensor Core 的低精度功能的方法。 它们被视为预览功能，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。 此功能可通过 `nvcuda::wmma::experimental` 命名空间获得：
```C++
namespace experimental { 
    namespace precision { 
        struct u4; // 4-bit unsigned 
        struct s4; // 4-bit signed 
        struct b1; // 1-bit 
   } 
    enum bmmaBitOp {
        bmmaBitOpXOR = 1, // compute_75 minimum
        bmmaBitOpAND = 2  // compute_80 minimum
    };
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 }; 
} 
```

对于 4 位精度，可用的 API 保持不变，但您必须指定 `experimental::precision::u4` 或 `experimental::precision::s4` 作为片段数据类型。 由于片段的元素被打包在一起，`num_storage_elements` 将小于该片段的 `num_elements`。 Sub-byte片段的 `num_elements` 变量，因此返回`Sub-byte`类型 `element_type<T>` 的元素数。 对于单位精度也是如此，在这种情况下，从 `element_type<T>` 到 `storage_element_type<T>` 的映射如下：
```C++
experimental::precision::u4 -> unsigned (8 elements in 1 storage element) 
experimental::precision::s4 -> int (8 elements in 1 storage element) 
experimental::precision::b1 -> unsigned (32 elements in 1 storage element) 
T -> T  //all other types
```

Sub-byte片段的允许布局始终为 `matrix_a` 的 `row_major` 和 `matrix_b `的 `col_major`。

对于子字节操作，`load_matrix_sync` 中 `ldm` 的值对于元素类型 `experimental::precision::u4` 和 `Experimental::precision::s4` 应该是 32 的倍数，或者对于元素类型 `experimental::precision::b1` 应该是 128 的倍数 （即，两种情况下都是 16 字节的倍数）。

`bmma_sync`:
等到所有warp lane都执行了`bmma_sync`，然后执行warp同步位矩阵乘法累加运算`D = (A op B) + C`，其中op由逻辑运算`bmmaBitOp`和`bmmaAccumulateOp`定义的累加组成。 可用的操作有：
* `bmmaBitOpXOR`，`matrix_a` 中的一行与 `matrix_b` 的 128 位列的 128 位 XOR
* `bmmaBitOpAND`，`matrix_a` 中的一行与 `matrix_b` 的 128 位列的 128 位 AND，可用于计算能力 8.0 及更高版本的设备。

累积操作始终是 `bmmaAccumulateOpPOPC`，它计算设置位的数量。

##    5. Restrictions
对于每个主要和次要设备架构，tensor cores所需的特殊格式可能不同。 由于线程仅持有整个矩阵的片段（不透明的架构特定的 ABI 数据结构），因此开发人员不允许对如何将各个参数映射到参与矩阵乘法累加的寄存器做出假设，这使情况变得更加复杂。

由于片段是特定于体系结构的，如果函数已针对不同的链接兼容体系结构编译并链接在一起成为相同的设备可执行文件，则将它们从函数 A 传递到函数 B 是不安全的。 在这种情况下，片段的大小和布局将特定于一种架构，而在另一种架构中使用 `WMMA API` 将导致不正确的结果或潜在的损坏。

片段布局不同的两个链接兼容架构的示例是 sm_70 和 sm_75。
```C++
fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }  
```
```C
// sm_70 fragment layout
$> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
// sm_75 fragment layout
$> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
// Linking the two together
$> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o   
```
这种未定义的行为在编译时和运行时的工具也可能无法检测到，因此需要格外小心以确保片段的布局是一致的。 当与既为不同的链接兼容架构构建并期望传递 WMMA 片段的遗留库链接时，最有可能出现这种链接危险。

请注意，在弱链接的情况下（例如，CUDA C++ 内联函数），链接器可能会选择任何可用的函数定义，这可能会导致编译单元之间的隐式传递。

为避免此类问题，矩阵应始终存储到内存中以通过外部接口传输（例如 `wmma::store_matrix_sync(dst, ...)`;），然后可以安全地将其作为指针类型传递给 `bar()` [ 例如 `float *dst`]。

请注意，由于 sm_70 可以在 sm_75 上运行，因此可以将上述示例 sm_75 代码更改为 sm_70 并在 sm_75 上正确运行。 但是，当与其他 sm_75 单独编译的二进制文件链接时，建议在您的应用程序中包含 sm_75 本机代码。

##    6. Element Types & Matrix Sizes
张量核心支持多种元素类型和矩阵大小。 下表显示了支持的 `matrix_a、matrix_b` 和`accumulator`矩阵的各种组合：
|Matrix A|	Matrix B|	Accumulator	|Matrix Size (m-n-k)|
|----|----|----|----|
|__half	|__half|	float|	16x16x16|
|__half|	__half|	float|	32x8x16|
|__half	|__half	|float|	8x32x16|
|__half	|__half	|__half	|16x16x16|
|__half	|__half	|__half	|32x8x16|
|__half	|__half	|__half	|8x32x16|
|unsigned char	|unsigned char|	int|	16x16x16|
|unsigned char	|unsigned char|	int	|32x8x16|
|unsigned char	|unsigned char|	int	|8x32x16|
|signed char	|signed char|	int	|16x16x16|
|signed char	|signed char|	int	|32x8x16|
|signed char	|signed char|	int	|8x32x16|

备用浮点支持：

|Matrix A	|Matrix B|	Accumulator|	Matrix Size (m-n-k)|
|----|----|----|----|
|__nv_bfloat16|	__nv_bfloat16|	float|	16x16x16|
|__nv_bfloat16|	__nv_bfloat16|	float	|32x8x16|
|__nv_bfloat16|	__nv_bfloat16|	float|	8x32x16|
|precision::tf32|	precision::tf32	|float|	16x16x8|

双精支持:

|Matrix A	|Matrix B	|Accumulator|	Matrix Size (m-n-k)|
|----|----|----|----|
|double|	double|	double|	8x8x4|

对sub-byte操作的实验性支持：

|Matrix A	|Matrix B	|Accumulator	|Matrix Size (m-n-k)|
|----|----|----|----|
|precision::u4|	precision::u4|	int|	8x8x32|
|precision::s4|	precision::s4|	int|	8x8x32|
|precision::b1|	precision::b1|	int	|8x8x128|

##    7. Example
以下代码在单个warp中实现 16x16x16 矩阵乘法:
```C++
#include <mma.h>
using namespace nvcuda;
      
__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}   
```
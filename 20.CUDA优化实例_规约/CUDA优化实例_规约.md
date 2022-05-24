# CUDA实例系列三:利用GPU优化向量规约问题

先简单的描述一下题目中说的向量规约问题. 

这里举个例子, 比如: 
* 我要求出1+2+3....+100的和
* 我要求出1*2*3....*100的积
* 我要找到a[100]中所有元素的最大值
* 我要找到a[100]中所有元素的最小值

诸如上边的问题, 我们可以简单的将其分解:
1 op 2 op 3 op 4.....op 100

这里的`op`代表一种操作, 操作的结果不会被顺序影响.

这时, 我们就可以将其分解为:

(1 op 2) op (3 op 4).....op (99 op 100)

所以我们就可以同时利用很多线程, 在一个时刻来计算所得括号中的内容.

接下来, 我们来看个实例:

我们利用CUDA来计算向量中所有元素的和

![](%E5%B9%BB%E7%81%AF%E7%89%871.JPG)

----

在上面的示例中, 我们要计算一个`a[32]`向量中所有元素的和,那么我们只需要按照以下步骤:

1. 每个线程从global memory中读取数据, 并将数据写到Shared memory中. 注意,这里的难点在于下图中的for循环. 这里的for循环是为了防止我们要处理的数据的数量远远大于我们能申请的线程的数量.就是利用CUDA中常用的grid-loop方法解决线程数少于数据数量的情况.

   **注意:大家千万别忽略了同步的步骤**

![](%E5%B9%BB%E7%81%AF%E7%89%872.JPG)

----

2. 接下来的for循环每个迭代步骤使用一些线程来计算数据的和.**注意:这里每个迭代步骤会相对于上一个迭代步骤使用的线程数量会减半**

![](%E5%B9%BB%E7%81%AF%E7%89%873.JPG)

----

3. 这里将每个block计算的结果放在global memory中的输出向量内. **注意:这里之所以这么做, 是因为我们CUDA在global memory中做同步操作代价非常大.当然也可以使用原子操作, 但是在有些情况下, 原子操作并不是最优解.它会产生等待的开销**

![](%E5%B9%BB%E7%81%AF%E7%89%874.JPG)

----

4. 这里做了两步, 就是为了避免使用原子操作

![](%E5%B9%BB%E7%81%AF%E7%89%875.JPG)

----


**接下来, 上源码,** (**特别提示: 写代码的时候比较飘逸, 请大家忽略不规范的命名规则**):


```C++
#include<stdio.h>
#include<stdint.h>
#include<time.h>     //for time()
#include<stdlib.h>   //for srand()/rand()
#include<sys/time.h> //for gettimeofday()/struct timeval


#define KEN_CHECK(r) \
{\
    cudaError_t rr = r;   \
    if (rr != cudaSuccess)\
    {\
        fprintf(stderr, "CUDA Error %s, function: %s, line: %d\n",       \
		        cudaGetErrorString(rr), __FUNCTION__, __LINE__); \
        exit(-1);\
    }\
}

#define N 10000000
#define BLOCK_SIZE 256
#define BLOCKS ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) //try next line if you can
//#define BLOCKS 666

__managed__ int source[N];               //input data
__managed__ int _partial_results[BLOCKS];//for 2-pass kernel
__managed__ int final_result[1] = {0};   //scalar output


__global__ void _hawk_sum_gpu(int *input, int count, int *output)
{
    __shared__ int bowman[BLOCK_SIZE];

    //**********register summation stage***********
    int komorebi = 0;
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
         idx < count;
	 idx += gridDim.x * blockDim.x
	)
    {
        komorebi += input[idx];
    }

    bowman[threadIdx.x] = komorebi;  //the per-thread partial sum is komorebi!
    __syncthreads();

    //**********shared memory summation stage***********
    for (int length = BLOCK_SIZE / 2; length >= 1; length /= 2)
    {
        int double_kill = -1;
	if (threadIdx.x < length)
	{
	    double_kill = bowman[threadIdx.x] + bowman[threadIdx.x + length];
	}
	__syncthreads();  //why we need two __syncthreads() here, and,
	
	if (threadIdx.x < length)
	{
	    bowman[threadIdx.x] = double_kill;
	}
	__syncthreads();  //....here ?
	
    } //the per-block partial sum is bowman[0]

    if (blockDim.x * blockIdx.x < count) //in case that our users are naughty
    {
        //per-block result written back, by thread 0, on behalf of a block.
        if (threadIdx.x == 0) output[blockIdx.x] = bowman[0];
    }
}

int _hawk_sum_cpu(int *ptr, int count)
{
    int sum = 0;
    for (int i = 0; i < count; i++)
    {
        sum += ptr[i];
    }
    return sum;
}

void _nanana_init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}

int main()
{
    //**********************************
    fprintf(stderr, "nanana is filling the buffer with %d elements...\n", N);
    _nanana_init(source, N);

    //**********************************
    //Now we are going to kick start your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!
    //Good luck & have fun!
    
    fprintf(stderr, "Running on GPU...\n");
    
double t0 = get_time();
    _hawk_sum_gpu<<<BLOCKS, BLOCK_SIZE>>>(source, N, _partial_results);
        KEN_CHECK(cudaGetLastError());  //checking for launch failures
	
    _hawk_sum_gpu<<<1, BLOCK_SIZE>>>(_partial_results, BLOCKS, final_result);
        KEN_CHECK(cudaGetLastError());  //the same
	
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
double t1 = get_time();

    int A = final_result[0];
    fprintf(stderr, "GPU sum: %u\n", A);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

double t2 = get_time();
    int B = _hawk_sum_cpu(source, N);
double t3 = get_time();
    fprintf(stderr, "CPU sum: %u\n", B);

    //******The last judgement**********
    if (A == B)
    {
        fprintf(stderr, "Test Passed!\n");
    }
    else
    {
        fprintf(stderr, "Test failed!\n");
	exit(-1);
    }
    
    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    return 0;
}	
	

```





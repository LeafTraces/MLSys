#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// a, b, c 都指向 GPU 内存；n 是元素个数
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    // TODO 1（你写）：算出全局下标 i，做越界保护，完成 c[i] = a[i] + b[i]
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1000;                      // 元素个数
    const size_t bytes = N * sizeof(float);  // 字节数 = 1000 * 4 = 4000

    // 1. host 端分配并赋值
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = 2.0f * i; }

    // 2. GPU 端分配（d_ 前缀表示 device）
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // 3. 输入从 host 拷到 device：参数顺序是（目的地, 来源, 字节数, 方向）
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;  // TODO 2（你写）：只用整数运算实现向上取整
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaError_t err = cudaDeviceSynchronize();  // 等 kernel 跑完，并拿到错误码
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

    // 5. 结果从 device 拷回 host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证：和 CPU 算的结果比最大误差
    float max_err = 0.0f;
    for (int i = 0; i < N; i++)
        max_err = fmaxf(max_err, fabsf(h_c[i] - (h_a[i] + h_b[i])));
    printf("blocks = %d, max_err = %g, c[999] = %g\n", blocks, max_err, h_c[N - 1]);

    // 7. 释放
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
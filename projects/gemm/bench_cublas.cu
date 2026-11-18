#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 计时一次 SGEMM（N×N 乘 N×N），返回秒
double time_sgemm(
    cublasHandle_t h, 
    int N, 
    const float* dA, 
    const float* dB, 
    float* dC
) {
    const float alpha = 1.0f, beta = 0.0f;

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    // cuBLAS 按列主序存矩阵。我们用行主序，所以把 A、B 交换传入，
    // 这样按行主序读出来的 dC 正好是 A·B
    cublasSgemm(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N, 
        N, N, N,
        &alpha, 
        dB, N, 
        dA, N, 
        &beta, 
        dC, N
        );
        
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    cublasHandle_t h;
    cublasCreate(&h);

    const int sizes[] = {512, 1024, 2048, 4096, 8192};
    const int warmup = 3, reps = 10;

    printf("%6s %12s %10s\n", "N", "median(ms)", "GFLOPS");
    for (int N : sizes) {
        size_t n_elem = (size_t)N * N;            // 元素个数
        size_t bytes  = n_elem * sizeof(float);   // 字节数

        std::vector<float> hA(n_elem), hB(n_elem);
        for (size_t i = 0; i < n_elem; i++) {
            hA[i] = rand() / (float)RAND_MAX;
            hB[i] = rand() / (float)RAND_MAX;
        }

        float *dA, *dB, *dC;
        cudaMalloc((void**)&dA, bytes);
        cudaMalloc((void**)&dB, bytes);
        cudaMalloc((void**)&dC, bytes);
        cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

        for (int i = 0; i < warmup; i++) time_sgemm(h, N, dA, dB, dC);

        std::vector<double> t(reps);
        for (int i = 0; i < reps; i++) t[i] = time_sgemm(h, N, dA, dB, dC);
        std::sort(t.begin(), t.end());
        
        double median = 0.0;
        if (reps % 2){
            median = t[reps / 2];
        }
        else{
            median = (t[reps / 2 - 1] + t[reps / 2]) / 2.0; // 单位：秒
        }

        double flops = 2.0 * N * N * N;
        double gflops = flops / median / 1e9;  // 用 N 和 median 算出 GFLOPS

        printf("%6d %12.3f %10.1f\n", 
            N, median * 1e3, gflops);

        cudaFree(dA); 
        cudaFree(dB); 
        cudaFree(dC);
    }
    cublasDestroy(h);
    return 0;
}
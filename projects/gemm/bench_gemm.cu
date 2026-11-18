#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error: %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ===== v1 朴素 GEMM =====
// A, B, C 都是 N×N、行主序、在 GPU 上；计算 C = A·B
__global__ void sgemm_naive(int N, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N){
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < N; k++){
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void launch_naive(int N, const float* A, const float* B, float* C) {
    dim3 block(32, 32);                          // 每个 block 32×32 = 1024 个线程
    dim3 grid((N + 31) / 32, (N + 31) / 32);     // 向上取整
    sgemm_naive<<<grid, block>>>(N, A, B, C);
}

void launch_cublas(cublasHandle_t h, int N, const float* A, const float* B, float* C) {
    const float alpha = 1.0f, beta = 0.0f;
    // 行主序技巧同 bench_cublas.cu：交换 A、B
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N);
}

// 用 CUDA event 计时：两个 event 放进 GPU 的任务队列，测的是 GPU 执行这两点之间的时间
template <typename F>
float time_ms(F f) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    f();
    cudaEventRecord(e);
    cudaEventSynchronize(e);                     // 等 GPU 执行到 e
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    int n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0f;
}

int main() {
    cublasHandle_t h;
    cublasCreate(&h);
    const int sizes[] = {512, 1024, 2048, 4096};  // 朴素版在 8192 太慢，先不测
    const int warmup = 2, reps = 10;

    printf("%8s %6s | %10s %9s | %10s %9s | %8s %12s\n",
           "time", "N", "cuBLAS ms", "GFLOPS", "naive ms", "GFLOPS", "%cuBLAS", "max_rel_err");

    for (int N : sizes) {
        size_t n_elem = (size_t)N * N;
        size_t bytes  = n_elem * sizeof(float);

        // 用简单公式生成 [0, 1) 的数据，比 rand() 快很多
        std::vector<float> hA(n_elem), hB(n_elem), hC(n_elem), hRef(n_elem);
        for (size_t i = 0; i < n_elem; i++) {
            hA[i] = ((i * 2654435761u) % 1000) / 1000.0f;
            hB[i] = ((i * 40503u + 7) % 1000) / 1000.0f;
        }

        float *dA, *dB, *dC, *dRef;
        CHECK(cudaMalloc((void**)&dA, bytes));
        CHECK(cudaMalloc((void**)&dB, bytes));
        CHECK(cudaMalloc((void**)&dC, bytes));
        CHECK(cudaMalloc((void**)&dRef, bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        // 正确性检查：以 cuBLAS 的结果为参考
        CHECK(cudaMemset(dC, 0, bytes));
        launch_cublas(h, N, dA, dB, dRef);
        launch_naive(N, dA, dB, dC);
        CHECK(cudaGetLastError());               // 检查 kernel 启动是否出错
        CHECK(cudaDeviceSynchronize());          // 检查 kernel 执行是否出错
        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f, max_ref = 0.0f;
        for (size_t i = 0; i < n_elem; i++) {
            max_diff = fmaxf(max_diff, fabsf(hC[i] - hRef[i]));
            max_ref  = fmaxf(max_ref, fabsf(hRef[i]));
        }
        float rel_err = max_diff / max_ref;

        // 预热
        for (int i = 0; i < warmup; i++) {
            launch_cublas(h, N, dA, dB, dRef);
            launch_naive(N, dA, dB, dC);
        }
        CHECK(cudaDeviceSynchronize());

        // 交替计时
        std::vector<float> t_cublas, t_naive;
        for (int i = 0; i < reps; i++) {
            t_cublas.push_back(time_ms([&] { launch_cublas(h, N, dA, dB, dRef); }));
            t_naive.push_back(time_ms([&] { launch_naive(N, dA, dB, dC); }));
        }
        float mc = median(t_cublas), mn = median(t_naive);
        double flop = 2.0 * N * N * N;
        double g_c = flop / (mc * 1e-3) / 1e9;
        double g_n = flop / (mn * 1e-3) / 1e9;

        char ts[16];
        time_t now = time(nullptr);
        strftime(ts, sizeof(ts), "%H:%M:%S", gmtime(&now));  // UTC，和 nvidia-smi 日志一致
        printf("%8s %6d | %10.3f %9.1f | %10.3f %9.1f | %7.2f%% %12.2e%s\n",
               ts, N, mc, g_c, mn, g_n, 100.0 * g_n / g_c, rel_err,
               rel_err < 1e-3f ? "" : "  <-- WRONG");

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    }
    cublasDestroy(h);
    return 0;
}
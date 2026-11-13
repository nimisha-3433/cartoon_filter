// Comparing CPU and GPU performance
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(x)  do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); std::exit(1); } } while (0)
#define CUDNN_CHECK(x) do { cudnnStatus_t e = (x); if (e != CUDNN_STATUS_SUCCESS) { \
    std::printf("cuDNN error: %s (line %d)\n", cudnnGetErrorString(e), __LINE__); std::exit(1); } } while (0)

// ------------------------------------------------------------------ settings
constexpr int   K        = 9;     // blur window size (9x9)
constexpr int   R        = K / 2;
constexpr float SIGMA    = 2.0f;  // blur strength
constexpr int   LEVELS   = 4;     // colours per channel
constexpr float EDGE     = 60.f;  // edge threshold
constexpr int   GPU_REPS = 50;    // runs averaged for the GPU versions
constexpr int   CPU_REPS = 3;     // runs averaged for the CPU version

// brightness weights for the B, G, R planes
constexpr float WB = 0.114f, WG = 0.587f, WR = 0.299f;
static const float SOBEL_X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
static const float SOBEL_Y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

float h_gauss[K * K];
__constant__ float c_gauss[K * K];

void make_gaussian() {
    float sum = 0;
    for (int y = -R; y <= R; ++y)
        for (int x = -R; x <= R; ++x)
            sum += h_gauss[(y + R) * K + (x + R)] = std::exp(-(x * x + y * y) / (2 * SIGMA * SIGMA));
    for (int i = 0; i < K * K; ++i) h_gauss[i] /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, h_gauss, sizeof(h_gauss)));
}

// Step 3, shared by all three versions: posterize + black edges for one pixel.
__host__ __device__ inline void finish_pixel(float b, float g, float r, float gx, float gy,
                                             unsigned char* out) {
    float step = 255.f / (LEVELS - 1);
    float keep = 1.f - fminf(fmaxf((sqrtf(gx * gx + gy * gy) - EDGE) / EDGE, 0.f), 1.f);
    out[0] = (unsigned char)(rintf(b / step) * step * keep + 0.5f);
    out[1] = (unsigned char)(rintf(g / step) * step * keep + 0.5f);
    out[2] = (unsigned char)(rintf(r / step) * step * keep + 0.5f);
}

// ================================================================== 1. CPU
int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void cpu_cartoon(const unsigned char* in, unsigned char* out, int W, int H) {
    const int N = W * H;
    std::vector<float> planes(3 * N), blur(3 * N);

    // BGRBGR... -> three separate planes (B, G, R)
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c) planes[c * N + i] = in[i * 3 + c];

    // 1. Gaussian blur
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float s = 0;
                for (int dy = -R; dy <= R; ++dy)
                    for (int dx = -R; dx <= R; ++dx)
                        s += h_gauss[(dy + R) * K + dx + R] *
                             planes[c * N + clampi(y + dy, 0, H - 1) * W + clampi(x + dx, 0, W - 1)];
                blur[c * N + y * W + x] = s;
            }

    // 2 + 3. Sobel on brightness, then posterize + edges
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int j = clampi(y + dy, 0, H - 1) * W + clampi(x + dx, 0, W - 1);
                    float L = WB * blur[j] + WG * blur[N + j] + WR * blur[2 * N + j];
                    gx += SOBEL_X[(dy + 1) * 3 + dx + 1] * L;
                    gy += SOBEL_Y[(dy + 1) * 3 + dx + 1] * L;
                }
            int i = y * W + x;
            finish_pixel(blur[i], blur[N + i], blur[2 * N + i], gx, gy, out + i * 3);
        }
}

// ================================================================== 2. CUDA (hand-written)
__global__ void to_planes(const uchar3* in, float* planes, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    planes[i] = in[i].x; planes[N + i] = in[i].y; planes[2 * N + i] = in[i].z;
}

__global__ void blur_kernel(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (x >= W || y >= H) return;
    const float* p = in + (size_t)c * W * H;
    float s = 0;
    for (int dy = -R; dy <= R; ++dy) {
        int yy = min(max(y + dy, 0), H - 1);
        for (int dx = -R; dx <= R; ++dx)
            s += c_gauss[(dy + R) * K + dx + R] * p[yy * W + min(max(x + dx, 0), W - 1)];
    }
    out[(size_t)c * W * H + y * W + x] = s;
}

__global__ void sobel_finish_kernel(const float* blur, unsigned char* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int N = W * H;
    const float sx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float sy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    float gx = 0, gy = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int j = min(max(y + dy, 0), H - 1) * W + min(max(x + dx, 0), W - 1);
            float L = WB * blur[j] + WG * blur[N + j] + WR * blur[2 * N + j];
            gx += sx[(dy + 1) * 3 + dx + 1] * L;
            gy += sy[(dy + 1) * 3 + dx + 1] * L;
        }
    int i = y * W + x;
    finish_pixel(blur[i], blur[N + i], blur[2 * N + i], gx, gy, out + i * 3);
}

// ================================================================== 3. cuDNN
__global__ void finish_kernel(const float* blur, const float* grad, unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    finish_pixel(blur[i], blur[N + i], blur[2 * N + i], grad[i], grad[N + i], out + i * 3);
}

struct Conv {
    cudnnTensorDescriptor_t x, y;
    cudnnFilterDescriptor_t f;
    cudnnConvolutionDescriptor_t c;
    cudnnConvolutionFwdAlgo_t algo;
    size_t ws_bytes = 0;
    float* w = nullptr;

    void setup(cudnnHandle_t h, int W, int H, int cin, int cout, int k, int groups,
               const std::vector<float>& weights) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&x));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&f));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&c));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(x, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, cin, H, W));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(y, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, cout, H, W));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(f, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, cout, cin / groups, k, k));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(c, k / 2, k / 2, 1, 1, 1, 1,
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(c, groups));
        CUDNN_CHECK(cudnnSetConvolutionMathType(c, CUDNN_FMA_MATH));

        cudnnConvolutionFwdAlgoPerf_t perf[8];
        int n = 0;
        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(h, x, f, c, y, 8, &n, perf));
        algo = perf[0].algo;
        ws_bytes = perf[0].memory;
        CUDA_CHECK(cudaMalloc(&w, weights.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(w, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void run(cudnnHandle_t h, const float* in, float* out, void* ws) {
        const float one = 1.f, zero = 0.f;
        CUDNN_CHECK(cudnnConvolutionForward(h, &one, x, in, f, w, c, algo, ws, ws_bytes, &zero, y, out));
    }

    void release() {
        cudnnDestroyTensorDescriptor(x); cudnnDestroyTensorDescriptor(y);
        cudnnDestroyFilterDescriptor(f); cudnnDestroyConvolutionDescriptor(c); cudaFree(w);
    }
};

// ================================================================== helpers
struct GpuTime { double kernels_ms = 0, total_ms = 0; };

template <class F>
GpuTime time_gpu(const unsigned char* h_in, unsigned char* h_out, uchar3* d_in, unsigned char* d_out,
                 size_t bytes, F work) {
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    GpuTime t;
    for (int rep = -5; rep < GPU_REPS; ++rep) {
        cudaEventRecord(e0);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);
        work();
        cudaEventRecord(e2);
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(e3);
        CUDA_CHECK(cudaEventSynchronize(e3));
        if (rep < 0) continue;
        float a, b;
        cudaEventElapsedTime(&a, e1, e2); t.kernels_ms += a;
        cudaEventElapsedTime(&b, e0, e3); t.total_ms += b;
    }
    t.kernels_ms /= GPU_REPS; t.total_ms /= GPU_REPS;
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    return t;
}

double match_percent(const cv::Mat& a, const cv::Mat& b) {
    int B = R + 2;
    cv::Rect inner(B, B, a.cols - 2 * B, a.rows - 2 * B);
    cv::Mat d;
    cv::absdiff(a(inner), b(inner), d);
    d = d.reshape(1, d.rows);
    return 100.0 * cv::countNonZero(d <= 1) / (double)d.total();
}

// ================================================================== main
int main(int argc, char** argv) {
    cv::Mat src;
    if (argc > 1) {
        src = cv::imread(argv[1]);
    } else {
        cv::VideoCapture cap(0);
        for (int i = 0; i < 15 && cap.isOpened(); ++i) cap.read(src);
    }
    if (src.empty()) { std::printf("No input: give an image path or connect a webcam\n"); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    make_gaussian();
    cudnnHandle_t dnn;
    CUDNN_CHECK(cudnnCreate(&dnn));

    // Prepare cuDNN weights
    std::vector<float> w_blur(3 * K * K), w_sobel(2 * 3 * 9);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < K * K; ++i) w_blur[c * K * K + i] = h_gauss[i];
    const float wc[3] = {WB, WG, WR};
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 9; ++i) {
            w_sobel[(0 * 3 + c) * 9 + i] = SOBEL_X[i] * wc[c];
            w_sobel[(1 * 3 + c) * 9 + i] = SOBEL_Y[i] * wc[c];
        }

    struct Res { const char* name; int w, h; } sizes[] = {
        {"480p", 854, 480}, {"720p", 1280, 720}, {"1080p", 1920, 1080}
    };

    std::printf("\nGPU: %s | cuDNN %zu | filter: %dx%d blur + Sobel + %d colour levels\n\n",
                prop.name, cudnnGetVersion(), K, K, LEVELS);
    std::printf("%-6s | %9s | %-22s | %-22s | %-15s | %s\n", "", "CPU", "CUDA  kernels / total",
                "cuDNN kernels / total", "speed-up vs CPU", "matches CPU");
    std::printf("-------+-----------+------------------------+------------------------+-----------------+------------------\n");

    for (const auto& s : sizes) {
        const int W = s.w, H = s.h, N = W * H;
        const size_t bytes = (size_t)N * 3;
        cv::Mat in;
        cv::resize(src, in, cv::Size(W, H), 0, 0, cv::INTER_AREA);
        cv::Mat out_cpu(H, W, CV_8UC3), out_cuda(H, W, CV_8UC3), out_dnn(H, W, CV_8UC3);

        // ---- 1. CPU
        double cpu_ms = 0;
        for (int rep = 0; rep < CPU_REPS; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            cpu_cartoon(in.data, out_cpu.data, W, H);
            cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        cpu_ms /= CPU_REPS;

        // ---- Allocate GPU memory
        unsigned char *h_in, *h_out, *d_out;
        uchar3* d_in;
        float *d_planes, *d_blur, *d_grad;
        CUDA_CHECK(cudaMallocHost(&h_in, bytes));
        CUDA_CHECK(cudaMallocHost(&h_out, bytes));
        std::memcpy(h_in, in.data, bytes);
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMalloc(&d_planes, 3 * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_blur, 3 * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, 2 * N * sizeof(float)));

        dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16), grd3((W + 15) / 16, (H + 15) / 16, 3);
        int lin = (N + 255) / 256;

        // ---- 2. CUDA
        GpuTime cuda = time_gpu(h_in, h_out, d_in, d_out, bytes, [&] {
            to_planes<<<lin, 256>>>(d_in, d_planes, N);
            blur_kernel<<<grd3, blk>>>(d_planes, d_blur, W, H);
            sobel_finish_kernel<<<grd, blk>>>(d_blur, d_out, W, H);
        });
        CUDA_CHECK(cudaGetLastError());
        std::memcpy(out_cuda.data, h_out, bytes);

        // ---- 3. cuDNN
        Conv blur, sobel;
        blur.setup(dnn, W, H, 3, 3, K, 3, w_blur);
        sobel.setup(dnn, W, H, 3, 2, 3, 1, w_sobel);
        void* ws = nullptr;
        size_t ws_bytes = std::max(blur.ws_bytes, sobel.ws_bytes);
        if (ws_bytes) CUDA_CHECK(cudaMalloc(&ws, ws_bytes));

        GpuTime dnnt = time_gpu(h_in, h_out, d_in, d_out, bytes, [&] {
            to_planes<<<lin, 256>>>(d_in, d_planes, N);
            blur.run(dnn, d_planes, d_blur, ws);
            sobel.run(dnn, d_blur, d_grad, ws);
            finish_kernel<<<lin, 256>>>(d_blur, d_grad, d_out, N);
        });
        CUDA_CHECK(cudaGetLastError());
        std::memcpy(out_dnn.data, h_out, bytes);

        // ---- Report results
        double m_cuda = match_percent(out_cpu, out_cuda);
        double m_dnn  = match_percent(out_cpu, out_dnn);
        std::printf("%-6s | %6.1f ms | %6.2f / %6.2f ms      | %6.2f / %6.2f ms      | %5.0fx / %5.0fx  | %5.1f%% / %5.1f%%\n",
                    s.name, cpu_ms, cuda.kernels_ms, cuda.total_ms, dnnt.kernels_ms, dnnt.total_ms,
                    cpu_ms / cuda.total_ms, cpu_ms / dnnt.total_ms, m_cuda, m_dnn);

        // Save outputs for 720p as a simple visual check
        if (H == 720) {
            cv::imwrite("out_cpu.png", out_cpu);
            cv::imwrite("out_cuda.png", out_cuda);
            cv::imwrite("out_cudnn.png", out_dnn);
        }

        // ---- Cleanup
        blur.release(); sobel.release();
        cudaFree(ws); cudaFree(d_in); cudaFree(d_out);
        cudaFree(d_planes); cudaFree(d_blur); cudaFree(d_grad);
        cudaFreeHost(h_in); cudaFreeHost(h_out);
    }

    cudnnDestroy(dnn);
    return 0;
}
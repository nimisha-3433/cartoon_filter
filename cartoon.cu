#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <chrono>

// Stop with a readable message if any CUDA call fails.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error: %s (line %d)\n",                  \
                         cudaGetErrorString(e), __LINE__);                      \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------- settings
struct Settings {
    int   levels      = 4;     // colours per channel after posterize
    float edge        = 60.f;  // Sobel threshold: lower = more outlines
    int   radius      = 4;     // bilateral window radius (pixels)
    int   passes      = 3;     // how many times the bilateral filter is applied
    float sigma_space = 3.f;   // how fast the weight falls with distance
    float sigma_color = 50.f;  // how different a colour may be and still be averaged
    float blend       = 0.8f;  // anti-flicker: 0.8 * new frame + 0.2 * previous
    int   style       = 2;     // colour style (1-6, see STYLES below); 2 = vivid
    float colourful   = 1.0f;  // extra saturation on top of the style (w / s keys)
};

// ---------------------------------------------------------------- colour styles
// saturation: 1 = unchanged, >1 = more colourful, 0 = grey.
// tint: multiplies blue, green, red. gray = 1 turns the image grey first (then tints it).
struct Style { const char* name; float saturation, tint_b, tint_g, tint_r; int gray; };
const Style STYLES[6] = {
    {"natural",       1.0f, 1.00f, 1.00f, 1.00f, 0},
    {"vivid",         1.6f, 1.00f, 1.00f, 1.00f, 0},
    {"warm",          1.4f, 0.85f, 1.00f, 1.12f, 0},
    {"cool",          1.4f, 1.15f, 1.00f, 0.88f, 0},
    {"black & white", 0.0f, 1.00f, 1.00f, 1.00f, 1},
    {"sepia",         0.0f, 0.70f, 0.90f, 1.10f, 1},
};

// ---------------------------------------------------------------- GPU kernels
// Each kernel runs one GPU thread per pixel.

// 1. Bilateral filter: average nearby pixels, but only those with a similar colour.
//    Flat areas become smooth, while edges stay sharp.
__global__ void bilateral(const uchar3* in, uchar3* out, int w, int h,
                          int r, float sigma_space, float sigma_color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uchar3 c = in[y * w + x];
    float sumB = 0, sumG = 0, sumR = 0, sumW = 0;
    float ks = -0.5f / (sigma_space * sigma_space);
    float kc = -0.5f / (sigma_color * sigma_color);

    for (int dy = -r; dy <= r; ++dy) {
        int yy = min(max(y + dy, 0), h - 1);           // clamp at the border
        for (int dx = -r; dx <= r; ++dx) {
            int xx = min(max(x + dx, 0), w - 1);
            uchar3 n = in[yy * w + xx];
            float db = (float)n.x - c.x, dg = (float)n.y - c.y, dr = (float)n.z - c.z;
            float wgt = __expf((dx * dx + dy * dy) * ks + (db * db + dg * dg + dr * dr) * kc);
            sumB += wgt * n.x; sumG += wgt * n.y; sumR += wgt * n.z; sumW += wgt;
        }
    }
    out[y * w + x] = make_uchar3(sumB / sumW + 0.5f, sumG / sumW + 0.5f, sumR / sumW + 0.5f);
}

// 2. Posterize + outlines in one pass.
//    Posterize: snap each colour channel to a few levels (flat cartoon colours).
//    Sobel: measure brightness change around the pixel; strong change = draw black.
__device__ float luma(uchar3 p) { return 0.114f * p.x + 0.587f * p.y + 0.299f * p.z; }

__global__ void cartoonize(const uchar3* smooth, uchar3* out, int w, int h,
                           int levels, float edge, Style st) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // colour style: push colours away from grey (saturation), then tint
    uchar3 p = smooth[y * w + x];
    float l = luma(p);
    float b = st.gray ? l : l + (p.x - l) * st.saturation;
    float g = st.gray ? l : l + (p.y - l) * st.saturation;
    float r = st.gray ? l : l + (p.z - l) * st.saturation;
    b = fminf(fmaxf(b * st.tint_b, 0.f), 255.f);
    g = fminf(fmaxf(g * st.tint_g, 0.f), 255.f);
    r = fminf(fmaxf(r * st.tint_r, 0.f), 255.f);

    // posterize
    float step = 255.f / (levels - 1);
    b = rintf(b / step) * step; g = rintf(g / step) * step; r = rintf(r / step) * step;

    // Sobel on the 3x3 neighbourhood
    auto L = [&](int dx, int dy) {
        int xx = min(max(x + dx, 0), w - 1), yy = min(max(y + dy, 0), h - 1);
        return luma(smooth[yy * w + xx]);
    };
    float gx = -L(-1, -1) - 2 * L(-1, 0) - L(-1, 1) + L(1, -1) + 2 * L(1, 0) + L(1, 1);
    float gy = -L(-1, -1) - 2 * L(0, -1) - L(1, -1) + L(-1, 1) + 2 * L(0, 1) + L(1, 1);
    float mag = sqrtf(gx * gx + gy * gy);

    // soft threshold: 0 below `edge`, fully black at 2*edge (smoother lines)
    float e = fminf(fmaxf((mag - edge) / edge, 0.f), 1.f);
    float keep = 1.f - e;
    out[y * w + x] = make_uchar3(b * keep, g * keep, r * keep);
}

// 3. Anti-flicker: mix the new frame with the previous output.
__global__ void blend(uchar3* cur, uchar3* prev, int n, float a, bool first) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uchar3 c = cur[i];
    if (!first) {
        uchar3 p = prev[i];
        c = make_uchar3(a * c.x + (1 - a) * p.x + 0.5f,
                        a * c.y + (1 - a) * p.y + 0.5f,
                        a * c.z + (1 - a) * p.z + 0.5f);
        cur[i] = c;
    }
    prev[i] = c;
}

// 4. BGR -> YUYV, the pixel format video-call apps expect from a webcam.
//    One thread handles two neighbouring pixels (they share U and V).
__global__ void bgr_to_yuyv(const uchar3* in, unsigned char* out, int w, int h) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    uchar3 a = in[y * w + x], b = in[y * w + x + 1];
    auto Y = [](uchar3 p) { return 16.f + 0.257f * p.z + 0.504f * p.y + 0.098f * p.x; };
    float B = (a.x + b.x) * 0.5f, G = (a.y + b.y) * 0.5f, R = (a.z + b.z) * 0.5f;
    float U = 128.f - 0.148f * R - 0.291f * G + 0.439f * B;
    float V = 128.f + 0.439f * R - 0.368f * G - 0.071f * B;
    unsigned char* o = out + (size_t)y * w * 2 + x * 2;
    o[0] = (unsigned char)fminf(fmaxf(Y(a), 0.f), 255.f);
    o[1] = (unsigned char)fminf(fmaxf(U, 0.f), 255.f);
    o[2] = (unsigned char)fminf(fmaxf(Y(b), 0.f), 255.f);
    o[3] = (unsigned char)fminf(fmaxf(V, 0.f), 255.f);
}

// ---------------------------------------------------------------- virtual camera
// Tell the v4l2loopback device which size/format we send, then write() frames into it.
int open_virtual_camera(const char* dev, int w, int h) {
    int fd = open(dev, O_RDWR);
    if (fd < 0) { std::perror("cannot open virtual camera (run scripts/setup_vcam.sh)"); return -1; }
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    ioctl(fd, VIDIOC_G_FMT, &fmt);
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = w * 2;
    fmt.fmt.pix.sizeimage = w * h * 2;
    fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { std::perror("VIDIOC_S_FMT"); close(fd); return -1; }
    std::printf("virtual camera: %s (%dx%d)\n", dev, w, h);
    return fd;
}

// ---------------------------------------------------------------- main loop
int main(int argc, char** argv) {
    const char* vcam_path = argc > 1 ? argv[1] : nullptr;
    const int W = 1280, H = 720, N = W * H;
    Settings s;

    // Webcam: ask for MJPG so the camera can deliver 720p at 30 FPS.
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) { std::fprintf(stderr, "cannot open webcam 0\n"); return 1; }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);

    int vcam = vcam_path ? open_virtual_camera(vcam_path, W, H) : -1;
    if (vcam_path && vcam < 0) return 1;

    // GPU memory: two images to ping-pong between, plus output/previous/YUYV.
    uchar3 *dA, *dB, *dOut, *dPrev;
    unsigned char* dYuyv;
    CUDA_CHECK(cudaMalloc(&dA, N * 3));
    CUDA_CHECK(cudaMalloc(&dB, N * 3));
    CUDA_CHECK(cudaMalloc(&dOut, N * 3));
    CUDA_CHECK(cudaMalloc(&dPrev, N * 3));
    CUDA_CHECK(cudaMalloc(&dYuyv, N * 2));

    // Pinned (page-locked) CPU memory makes the copies to/from the GPU faster.
    unsigned char *hIn, *hOut, *hYuyv;
    CUDA_CHECK(cudaMallocHost(&hIn, N * 3));
    CUDA_CHECK(cudaMallocHost(&hOut, N * 3));
    CUDA_CHECK(cudaMallocHost(&hYuyv, N * 2));
    cv::Mat out(H, W, CV_8UC3, hOut);

    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    dim3 gridYuyv((W / 2 + 15) / 16, (H + 15) / 16);
    bool first = true, show_text = true;
    double fps = 0;
    cv::Mat frame;

    std::printf("running - keys: +/- colours, [/] outlines, ./, smoothing, 1-6 colour style, w/s colourful, h text, q quit\n");
    while (true) {
        auto t0 = std::chrono::steady_clock::now();

        // 1. capture
        if (!cap.read(frame) || frame.empty()) break;
        if (frame.cols != W || frame.rows != H) cv::resize(frame, frame, cv::Size(W, H));
        std::memcpy(hIn, frame.data, N * 3);

        // 2. upload to GPU
        CUDA_CHECK(cudaMemcpy(dA, hIn, N * 3, cudaMemcpyHostToDevice));

        // 3. smooth several times (A -> B -> A ...)
        for (int i = 0; i < s.passes; ++i) {
            bilateral<<<grid, block>>>(dA, dB, W, H, s.radius, s.sigma_space, s.sigma_color);
            std::swap(dA, dB);
        }

        // 4. flat colours + black outlines
        Style st = STYLES[s.style - 1];
        st.saturation *= s.colourful;
        cartoonize<<<grid, block>>>(dA, dOut, W, H, s.levels, s.edge, st);

        // 5. anti-flicker
        blend<<<(N + 255) / 256, 256>>>(dOut, dPrev, N, s.blend, first);
        first = false;
        CUDA_CHECK(cudaGetLastError());

        // 6. download for the preview window
        CUDA_CHECK(cudaMemcpy(hOut, dOut, N * 3, cudaMemcpyDeviceToHost));

        // 7. convert + send to the virtual camera
        if (vcam >= 0) {
            bgr_to_yuyv<<<gridYuyv, block>>>(dOut, dYuyv, W, H);
            CUDA_CHECK(cudaMemcpy(hYuyv, dYuyv, N * 2, cudaMemcpyDeviceToHost));
            if (write(vcam, hYuyv, N * 2) != N * 2) std::fprintf(stderr, "virtual camera write failed\n");
        }

        // 8. preview window with FPS
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fps = 0.9 * fps + 0.1 * (1000.0 / ms);
        cv::Mat view = out.clone();
        if (show_text) {
            char line[200];
            std::snprintf(line, sizeof(line), "%.0f FPS | %s x%.1f | colours %d | edge %.0f | radius %d",
                          fps, STYLES[s.style - 1].name, s.colourful, s.levels, s.edge, s.radius);
            cv::putText(view, line, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 0}, 4);
            cv::putText(view, line, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);
        }
        cv::imshow("cartoon (simple)", view);

        // 9. keys
        int k = cv::waitKey(1) & 0xFF;
        if (k == 'q' || k == 27) break;
        if ((k == '+' || k == '=') && s.levels < 16) s.levels++;
        if (k == '-' && s.levels > 2) s.levels--;
        if (k == ']') s.edge += 10;
        if (k == '[' && s.edge > 10) s.edge -= 10;
        if (k == '.' && s.radius < 8) s.radius++;
        if (k == ',' && s.radius > 1) s.radius--;
        if (k == 'h') show_text = !show_text;
        if (k >= '1' && k <= '6') s.style = k - '0';
        if (k == 'w' && s.colourful < 2.5f) s.colourful += 0.1f;
        if (k == 's' && s.colourful > 0.15f) s.colourful -= 0.1f;
    }

    if (vcam >= 0) close(vcam);
    cudaFree(dA); cudaFree(dB); cudaFree(dOut); cudaFree(dPrev); cudaFree(dYuyv);
    cudaFreeHost(hIn); cudaFreeHost(hOut); cudaFreeHost(hYuyv);
    return 0;
}

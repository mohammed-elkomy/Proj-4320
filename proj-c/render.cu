/*  render.cu  —  CUDA triangle rasteriser
 *
 *  Drop-in GPU replacement for render.c.
 *  Exports the same render_triangles() signature so ga.c stays unchanged,
 *  plus two lifecycle calls the caller must bookend around the GA run:
 *
 *      cuda_renderer_init(w, h);   // once, after loading the target image
 *      ...run GA...
 *      cuda_renderer_free();       // once, before exit
 *
 *  Kernel design
 *  ─────────────
 *  One CUDA thread per pixel.  Each thread iterates over all N_TRIANGLES in
 *  draw order, performs the same barycentric test + alpha blend as render.c,
 *  and writes its final RGB value directly to the output buffer.
 *
 *  Triangle data (vertices + colors) is small:
 *      N_TRIANGLES * (6 verts + 4 colors) * 4 bytes = 200 * 10 * 4 = 8 KB
 *  This fits comfortably in shared memory, so every block loads it once and
 *  all threads read from fast on-chip storage for the inner loop.
 *
 *  Device buffers are allocated once in cuda_renderer_init() and reused on
 *  every render call to avoid per-call cudaMalloc overhead.
 */

#include "triangle_ga.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/* ── Error-checking wrapper ──────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ── Persistent device buffers (allocated once, reused every render call) ── */
static float *d_verts  = NULL;   /* [N_VERTEX_GENES]   */
static float *d_colors = NULL;   /* [N_COLOR_GENES]    */
static float *d_canvas = NULL;   /* [H * W * 3] floats */


/* ════════════════════════════════════════════════════════════════════════════
 *  Kernel
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  Grid : ceil(W/16) × ceil(H/16) blocks of 16×16 threads.
 *  Each thread = one output pixel (px, py).
 *
 *  Shared memory layout: [ sv: N_VERTEX_GENES floats | sc: N_COLOR_GENES floats ]
 *  All threads in the block cooperate to load this data before the inner loop.
 */
__global__ void render_kernel(
    const float * __restrict__ verts,    /* [N_TRIANGLES * 6]  normalised [0,1] */
    const float * __restrict__ colors,   /* [N_TRIANGLES * 4]  RGBA             */
    float       * __restrict__ canvas,   /* [H * W * 3]        output           */
    int w, int h)
{
    extern __shared__ float sdata[];
    float *sv = sdata;                   /* vertex data  */
    float *sc = sdata + N_VERTEX_GENES;  /* color data   */

    int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x  * blockDim.y;

    /* Cooperatively load all triangle data into shared memory */
    for (int i = tid; i < N_VERTEX_GENES; i += block_size) sv[i] = verts[i];
    for (int i = tid; i < N_COLOR_GENES;  i += block_size) sc[i] = colors[i];
    __syncthreads();

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float cr = 1.0f, cg = 1.0f, cb = 1.0f;   /* white background */

    for (int t = 0; t < N_TRIANGLES; t++) {
        /* Scale normalised [0,1] vertex coords to pixel space */
        float x0 = sv[t*6 + 0] * w,  y0 = sv[t*6 + 1] * h;
        float x1 = sv[t*6 + 2] * w,  y1 = sv[t*6 + 3] * h;
        float x2 = sv[t*6 + 4] * w,  y2 = sv[t*6 + 5] * h;

        float tr = sc[t*4 + 0];
        float tg = sc[t*4 + 1];
        float tb = sc[t*4 + 2];
        float ta = sc[t*4 + 3];

        /* Edge vectors — same convention as render.c / Python:
         *   v0 = tri[2] - tri[0],   v1 = tri[1] - tri[0]          */
        float v0x = x2 - x0,  v0y = y2 - y0;
        float v1x = x1 - x0,  v1y = y1 - y0;

        float d00 = v0x*v0x + v0y*v0y;
        float d01 = v0x*v1x + v0y*v1y;
        float d11 = v1x*v1x + v1y*v1y;

        float denom = d00*d11 - d01*d01;
        if (fabsf(denom) < 1e-10f) continue;   /* degenerate triangle */
        float inv = 1.0f / denom;

        float v2x = (float)px - x0;
        float v2y = (float)py - y0;

        float d02 = v0x*v2x + v0y*v2y;
        float d12 = v1x*v2x + v1y*v2y;

        float u = (d11*d02 - d01*d12) * inv;
        float v = (d00*d12 - d01*d02) * inv;

        if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
            float one_minus_a = 1.0f - ta;
            cr = tr*ta + cr*one_minus_a;
            cg = tg*ta + cg*one_minus_a;
            cb = tb*ta + cb*one_minus_a;
        }
    }

    int idx = (py * w + px) * 3;
    canvas[idx + 0] = cr;
    canvas[idx + 1] = cg;
    canvas[idx + 2] = cb;
}


/* ════════════════════════════════════════════════════════════════════════════
 *  Public API  (extern "C" so C callers in ga.c / main.c link correctly)
 * ════════════════════════════════════════════════════════════════════════════ */

extern "C" void cuda_renderer_init(int w, int h)
{
    CUDA_CHECK(cudaMalloc(&d_verts,  N_VERTEX_GENES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors, N_COLOR_GENES  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_canvas, w * h * 3      * sizeof(float)));

    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("[CUDA] Device  : %s  (compute %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("[CUDA] Buffers : %.1f KB verts  %.1f KB colors  %.1f KB canvas\n",
           N_VERTEX_GENES * sizeof(float) / 1024.0f,
           N_COLOR_GENES  * sizeof(float) / 1024.0f,
           w * h * 3      * sizeof(float) / 1024.0f);
}

extern "C" void cuda_renderer_free(void)
{
    cudaFree(d_verts);   d_verts  = NULL;
    cudaFree(d_colors);  d_colors = NULL;
    cudaFree(d_canvas);  d_canvas = NULL;
}

/*  render_triangles — same signature as render.c, now GPU-accelerated.
 *
 *  Hot path (called ~DISC_COUNT times per GA generation):
 *    1. Upload verts + colors to GPU   (~8 KB each direction, fast)
 *    2. Launch kernel — all pixels in parallel
 *    3. Download rendered canvas back  (W*H*3*4 bytes — main transfer cost)
 *
 *  The MSE loop in ga.c still runs on CPU.  Moving it to the GPU with a
 *  reduction kernel would eliminate step 3 entirely and is the natural
 *  next optimisation if the transfer becomes the bottleneck.
 */
extern "C" void render_triangles(const float *vertices_flat, const float *colors,
                                 Image *dst, Profiler *prof)
{
    double t0 = profiler_now();

    const int w = dst->w;
    const int h = dst->h;

    /* Upload triangle data (~8 KB total — negligible) */
    CUDA_CHECK(cudaMemcpy(d_verts,  vertices_flat,
                          N_VERTEX_GENES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors, colors,
                          N_COLOR_GENES  * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch: 16×16 thread blocks, shared memory holds all triangle data */
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    size_t smem = (N_VERTEX_GENES + N_COLOR_GENES) * sizeof(float);

    render_kernel<<<grid, block, smem>>>(d_verts, d_colors, d_canvas, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download rendered canvas back to CPU */
    CUDA_CHECK(cudaMemcpy(dst->data, d_canvas,
                          w * h * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    profiler_add(prof, BUCKET_RENDER, profiler_now() - t0);
}

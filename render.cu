/*  render.cu  —  CUDA triangle rasteriser + GPU batch MSE evaluator
 *
 *  Two rendering paths:
 *
 *  1. render_triangles()        — single chromosome, downloads canvas to CPU
 *                                 Used only for PPM snapshot saves.
 *
 *  2. batch_compute_loss_gpu()  — N chromosomes, computes MSE entirely on GPU.
 *                                 Returns only N scalar loss values (~N*4 bytes).
 *                                 Used for every GA fitness evaluation.
 *
 *  Device buffers (persistent, allocated once in cuda_renderer_init):
 *    d_verts       [N_VERTEX_GENES]               — single-render verts
 *    d_colors      [N_COLOR_GENES]                — single-render colors
 *    d_canvas      [H*W*3]                        — single-render output
 *    d_target      [H*W*3]                        — target image (uploaded once)
 *    d_all_verts   [POP_SIZE * N_VERTEX_GENES]    — batch verts
 *    d_all_colors  [POP_SIZE * N_COLOR_GENES]     — batch colors
 *    d_losses      [POP_SIZE]                     — batch MSE output
 */

#include "app.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* ── Runtime sizes (set by cuda_renderer_init) ──────────────────────────── */
static int g_pop_size;
static int g_n_triangles;
static int g_n_vertex_genes;
static int g_n_color_genes;
static int g_n_genes;

/* ── Persistent device buffers ─────────────────────────────────────────── */
static float *d_verts      = NULL;
static float *d_colors     = NULL;
static float *d_canvas     = NULL;
static float *d_target     = NULL;
static float *d_all_verts  = NULL;
static float *d_all_colors = NULL;
static float *d_losses     = NULL;

/* ── Pinned host staging buffers (cudaMallocHost — DMA-able, no copy overhead) */
static float *h_all_verts  = NULL;
static float *h_all_colors = NULL;
static float *h_losses     = NULL;


/* ════════════════════════════════════════════════════════════════════════
 *  Kernel 1 — single render  (PPM snapshots only)
 *
 *  Grid : (ceil(W/16), ceil(H/16))   Block: (16, 16)
 *  One thread per pixel; triangle data loaded into shared memory.
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void render_kernel(
    const float * __restrict__ verts,
    const float * __restrict__ colors,
    float       * __restrict__ canvas,
    int w, int h, int n_triangles, int n_vertex_genes, int n_color_genes)
{
    extern __shared__ float sdata[];
    float *sv = sdata;
    float *sc = sdata + n_vertex_genes;

    int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x  * blockDim.y;

    for (int i = tid; i < n_vertex_genes; i += block_size) sv[i] = verts[i];
    for (int i = tid; i < n_color_genes;  i += block_size) sc[i] = colors[i];
    __syncthreads();

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    float cr = 1.f, cg = 1.f, cb = 1.f;

    for (int t = 0; t < n_triangles; t++) {
        float x0=sv[t*6+0]*w, y0=sv[t*6+1]*h;
        float x1=sv[t*6+2]*w, y1=sv[t*6+3]*h;
        float x2=sv[t*6+4]*w, y2=sv[t*6+5]*h;
        float tr=sc[t*4+0], tg=sc[t*4+1], tb=sc[t*4+2], ta=sc[t*4+3];

        float v0x=x2-x0, v0y=y2-y0, v1x=x1-x0, v1y=y1-y0;
        float d00=v0x*v0x+v0y*v0y, d01=v0x*v1x+v0y*v1y, d11=v1x*v1x+v1y*v1y;
        float denom=d00*d11-d01*d01;
        if (fabsf(denom)<1e-10f) continue;
        float inv=1.f/denom, v2x=(float)px-x0, v2y=(float)py-y0;
        float d02=v0x*v2x+v0y*v2y, d12=v1x*v2x+v1y*v2y;
        float u=(d11*d02-d01*d12)*inv, v=(d00*d12-d01*d02)*inv;
        if (u>=0&&v>=0&&u+v<=1) {
            float a1=1.f-ta;
            cr=tr*ta+cr*a1; cg=tg*ta+cg*a1; cb=tb*ta+cb*a1;
        }
    }

    int idx=(py*w+px)*3;
    canvas[idx]=cr; canvas[idx+1]=cg; canvas[idx+2]=cb;
}


/* ════════════════════════════════════════════════════════════════════════
 *  Kernel 2 — batch render + inline MSE  (GA fitness evaluations)
 *
 *  Grid : (ceil(W/16), ceil(H/16), count)   Block: (16, 16, 1)
 *  blockIdx.z = chromosome index.
 *
 *  Each thread renders its pixel for chromosome blockIdx.z, computes
 *  squared error vs target, then the block tree-reduces 256 values to 1
 *  and atomicAdds into losses[blockIdx.z].
 *
 *  No canvas download — only losses[] (count floats) is transferred back.
 *
 *  Shared memory layout (per block, ~8.8 KB):
 *    sv       [N_VERTEX_GENES]   — triangle vertices
 *    sc       [N_COLOR_GENES]    — triangle colors
 *    s_reduce [256]              — block reduction scratch
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void batch_render_mse_kernel(
    const float * __restrict__ all_verts,
    const float * __restrict__ all_colors,
    const float * __restrict__ target,
    float       *              losses,
    int w, int h, int n_triangles, int n_vertex_genes, int n_color_genes)
{
    int chrom = blockIdx.z;

    const float *verts  = all_verts  + chrom * n_vertex_genes;
    const float *colors = all_colors + chrom * n_color_genes;

    extern __shared__ float sdata[];
    float *sv       = sdata;
    float *sc       = sdata + n_vertex_genes;
    float *s_reduce = sdata + n_vertex_genes + n_color_genes;

    int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x  * blockDim.y;

    for (int i = tid; i < n_vertex_genes; i += block_size) sv[i] = verts[i];
    for (int i = tid; i < n_color_genes;  i += block_size) sc[i] = colors[i];
    __syncthreads();

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    float sq_err = 0.f;

    if (px < w && py < h) {
        float cr = 1.f, cg = 1.f, cb = 1.f;

        for (int t = 0; t < n_triangles; t++) {
            float x0=sv[t*6+0]*w, y0=sv[t*6+1]*h;
            float x1=sv[t*6+2]*w, y1=sv[t*6+3]*h;
            float x2=sv[t*6+4]*w, y2=sv[t*6+5]*h;
            float tr=sc[t*4+0], tg=sc[t*4+1], tb=sc[t*4+2], ta=sc[t*4+3];

            float v0x=x2-x0, v0y=y2-y0, v1x=x1-x0, v1y=y1-y0;
            float d00=v0x*v0x+v0y*v0y, d01=v0x*v1x+v0y*v1y, d11=v1x*v1x+v1y*v1y;
            float denom=d00*d11-d01*d01;
            if (fabsf(denom)<1e-10f) continue;
            float inv=1.f/denom, v2x=(float)px-x0, v2y=(float)py-y0;
            float d02=v0x*v2x+v0y*v2y, d12=v1x*v2x+v1y*v2y;
            float u=(d11*d02-d01*d12)*inv, v=(d00*d12-d01*d02)*inv;
            if (u>=0&&v>=0&&u+v<=1) {
                float a1=1.f-ta;
                cr=tr*ta+cr*a1; cg=tg*ta+cg*a1; cb=tb*ta+cb*a1;
            }
        }

        /* Squared error vs target for this pixel (R + G + B) */
        int idx=(py*w+px)*3;
        float dr=cr-target[idx], dg=cg-target[idx+1], db=cb-target[idx+2];
        sq_err = dr*dr + dg*dg + db*db;
    }

    /* Tree reduction: 256 → 1 within the block */
    s_reduce[tid] = sq_err;
    __syncthreads();
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }

    /* One atomic add per block — far fewer atomics than one per thread */
    if (tid == 0)
        atomicAdd(&losses[chrom], s_reduce[0]);
}


/* ════════════════════════════════════════════════════════════════════════
 *  Public API
 * ════════════════════════════════════════════════════════════════════════ */

extern "C" void cuda_renderer_init(const Image *target, const AppConfig *cfg)
{
    g_pop_size      = cfg->pop_size;
    g_n_triangles   = cfg->n_triangles;
    g_n_vertex_genes = cfg->n_vertex_genes;
    g_n_color_genes  = cfg->n_color_genes;
    g_n_genes        = cfg->n_genes;

    int w = target->w, h = target->h;

    CUDA_CHECK(cudaMalloc(&d_verts,  g_n_vertex_genes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors, g_n_color_genes  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_canvas, w * h * 3        * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_target, w * h * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_target, target->data,
                          w * h * 3 * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_all_verts,  (size_t)g_pop_size * g_n_vertex_genes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_colors, (size_t)g_pop_size * g_n_color_genes  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_losses,     (size_t)g_pop_size                    * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&h_all_verts,  (size_t)g_pop_size * g_n_vertex_genes * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_all_colors, (size_t)g_pop_size * g_n_color_genes  * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_losses,     (size_t)g_pop_size                    * sizeof(float)));

    int dev; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("[CUDA] Device  : %s  (compute %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("[CUDA] Config  : %d triangles  pop=%d  genes=%d\n",
           g_n_triangles, g_pop_size, g_n_genes);
    printf("[CUDA] Buffers : %.1f KB target  %.1f KB batch verts  %.1f KB batch colors\n",
           w*h*3*sizeof(float)/1024.f,
           (float)(g_pop_size*g_n_vertex_genes*sizeof(float))/1024.f,
           (float)(g_pop_size*g_n_color_genes *sizeof(float))/1024.f);
}

extern "C" void cuda_renderer_free(void)
{
    cudaFree(d_verts);      d_verts      = NULL;
    cudaFree(d_colors);     d_colors     = NULL;
    cudaFree(d_canvas);     d_canvas     = NULL;
    cudaFree(d_target);     d_target     = NULL;
    cudaFree(d_all_verts);  d_all_verts  = NULL;
    cudaFree(d_all_colors); d_all_colors = NULL;
    cudaFree(d_losses);     d_losses     = NULL;

    cudaFreeHost(h_all_verts);  h_all_verts  = NULL;
    cudaFreeHost(h_all_colors); h_all_colors = NULL;
    cudaFreeHost(h_losses);     h_losses     = NULL;
}

/* render_triangles — single render + canvas download, used for snapshots only */
extern "C" void render_triangles(const float *vertices_flat, const float *colors,
                                 Image *dst, Profiler *prof)
{
    double t0 = profiler_now();
    int w = dst->w, h = dst->h;

    CUDA_CHECK(cudaMemcpy(d_verts,  vertices_flat,
                          g_n_vertex_genes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors, colors,
                          g_n_color_genes  * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w+15)/16, (h+15)/16);
    size_t smem = (size_t)(g_n_vertex_genes + g_n_color_genes) * sizeof(float);

    render_kernel<<<grid, block, smem>>>(d_verts, d_colors, d_canvas, w, h,
                                         g_n_triangles, g_n_vertex_genes, g_n_color_genes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst->data, d_canvas,
                          w*h*3*sizeof(float), cudaMemcpyDeviceToHost));

    profiler_add(prof, BUCKET_RENDER, profiler_now() - t0);
}

/*
 * batch_compute_loss_gpu
 *
 * Renders `count` chromosomes and computes MSE vs the target image entirely
 * on the GPU. Only `count` floats are transferred back to the CPU.
 *
 * pop        : flat array — chromosome c starts at pop[c * N_GENES] (double)
 * count      : number of chromosomes to evaluate (≤ POP_SIZE)
 * losses_out : receives count MSE values as doubles
 * w, h       : image dimensions
 *
 * Transfer cost per call:
 *   Upload : count * (N_VERTEX_GENES + N_COLOR_GENES) * 4 bytes
 *   Download: count * 4 bytes   (vs count * W*H*3*4 bytes previously)
 */
extern "C" void batch_compute_loss_gpu(const double *pop, int count,
                                        double *losses_out, int w, int h,
                                        Profiler *prof)
{
    double t0 = profiler_now();

    for (int c = 0; c < count; c++) {
        const double *x      = pop          + (size_t)c * g_n_genes;
        float        *verts  = h_all_verts  + c * g_n_vertex_genes;
        float        *colors = h_all_colors + c * g_n_color_genes;
        for (int i = 0; i < g_n_vertex_genes; i++) verts[i]  = (float)x[i];
        for (int i = 0; i < g_n_color_genes;  i++) colors[i] = (float)x[g_n_vertex_genes + i];
    }

    CUDA_CHECK(cudaMemcpy(d_all_verts,  h_all_verts,
                          (size_t)count * g_n_vertex_genes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_colors, h_all_colors,
                          (size_t)count * g_n_color_genes  * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_losses, 0, count * sizeof(float)));

    dim3 block(16, 16, 1);
    dim3 grid((w+15)/16, (h+15)/16, count);
    size_t smem = (size_t)(g_n_vertex_genes + g_n_color_genes + 256) * sizeof(float);

    batch_render_mse_kernel<<<grid, block, smem>>>(
        d_all_verts, d_all_colors, d_target, d_losses, w, h,
        g_n_triangles, g_n_vertex_genes, g_n_color_genes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download only the loss scalars (~count * 4 bytes) */
    CUDA_CHECK(cudaMemcpy(h_losses, d_losses,
                          count * sizeof(float), cudaMemcpyDeviceToHost));

    /* Normalise to MSE: divide by total number of float elements */
    double n = (double)w * h * 3;
    for (int c = 0; c < count; c++)
        losses_out[c] = (double)h_losses[c] / n;

    if (prof) profiler_add(prof, BUCKET_RENDER, profiler_now() - t0);
}

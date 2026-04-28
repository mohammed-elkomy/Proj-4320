/*  render.cu  —  CUDA triangle rasteriser + multi-GPU batch loss evaluator
 *
 *  Two rendering paths:
 *
 *  1. render_triangles()        — single chromosome, downloads canvas to CPU.
 *                                 Runs on GPU 0 only.  Used for PPM snapshots.
 *
 *  2. batch_compute_loss_gpu()  — N chromosomes, computes loss entirely on GPU.
 *                                 Chromosomes are split evenly across all active
 *                                 GPUs; each GPU runs its slice in parallel via
 *                                 independent CUDA streams.  Only N loss scalars
 *                                 (~N×4 bytes) are transferred back to the host.
 *                                 Supports LOSS_MSE, LOSS_L4, LOSS_SSIM (app.h).
 *
 *  Multi-GPU memory layout (per GPU, set up in cuda_renderer_init):
 *    d_target      [H*W*3]                — target image replica (one per GPU)
 *    d_all_verts   [POP_SIZE*N_VG]        — batch vertices for this GPU's slice
 *    d_all_colors  [POP_SIZE*N_CG]        — batch colors   for this GPU's slice
 *    d_losses      [POP_SIZE]             — batch loss output
 *    h_losses      [POP_SIZE]  (pinned)   — per-GPU D2H staging for losses
 *
 *  GPU 0 additionally holds the single-render buffers:
 *    d_verts  [N_VG]  d_colors  [N_CG]  d_canvas  [H*W*3]
 *
 *  Shared host pinned input staging (one allocation, all GPUs DMA their slice):
 *    h_all_verts   [POP_SIZE * N_VERTEX_GENES]
 *    h_all_colors  [POP_SIZE * N_COLOR_GENES]
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

/* ── Per-GPU context ────────────────────────────────────────────────────── */
#define MAX_GPUS 8

typedef struct {
    int          device_id;
    cudaStream_t stream;
    /* Single-render buffers — GPU 0 only (PPM snapshots) */
    float       *d_verts;
    float       *d_colors;
    float       *d_canvas;
    /* Per-GPU device buffers for batch evaluation */
    float       *d_target;      /* full target image replica */
    float       *d_all_verts;   /* this GPU's chromosome slice — vertices */
    float       *d_all_colors;  /* this GPU's chromosome slice — colors   */
    float       *d_losses;      /* loss output for this GPU's slice        */
    /* Pinned per-GPU output staging (D2H, independent per GPU) */
    float       *h_losses;
} GpuContext;

static GpuContext g_gpus[MAX_GPUS];
static int        g_num_gpus = 0;

/* ── Shared pinned input staging (one copy, each GPU DMAs its own slice) ── */
static float *h_all_verts  = NULL;   /* POP_SIZE * N_VERTEX_GENES */
static float *h_all_colors = NULL;   /* POP_SIZE * N_COLOR_GENES  */

/*
 * Block dimensions
 *
 * RENDER_BLK / LOSS_BLK (32×32 = 1024 threads):
 *   Used for the single-render and MSE/L4 batch kernels.  1024 threads/block
 *   maximises warp count for latency hiding; shared mem is low (~12 KB),
 *   so occupancy is not shared-memory-bound.
 *
 * SSIM_BLK (16×16 = 256 threads):
 *   The SSIM kernel allocates 16 reduction arrays of block_size floats each.
 *   At 32×32 that is 16×1024×4 = 64 KB of scratch alone, pushing total shared
 *   mem to ~72 KB.  With only 96 KB per SM on Volta, only ONE block would fit
 *   per SM — 4× worse occupancy than 16×16 (which uses ~24 KB and fits four).
 *   Keeping SSIM at 16×16 preserves four-way SM concurrency and avoids
 *   needing cudaFuncSetAttribute to raise the shared-memory limit.
 */
#define RENDER_BLK 16   /* render_kernel and batch MSE/L4 */
#define SSIM_BLK   16   /* batch SSIM — shared-memory-limited */


/* ════════════════════════════════════════════════════════════════════════
 *  Kernel 1 — single render  (PPM snapshots only)
 *
 *  Grid : (ceil(W/RENDER_BLK), ceil(H/RENDER_BLK))   Block: (RENDER_BLK, RENDER_BLK)
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
 *  Per-pixel loss helpers (device-only)
 *
 *  pixel_loss_mse  — sum of squared channel errors (L2^2 per pixel)
 *  pixel_loss_l4   — sum of quartic channel errors  (L4   per pixel)
 *                    A pixel twice as wrong costs 16× instead of 4×;
 *                    this aggressively penalises the worst mistakes.
 * ════════════════════════════════════════════════════════════════════════ */
__device__ __forceinline__ float pixel_loss_mse(float dr, float dg, float db)
{
    return dr*dr + dg*dg + db*db;
}

__device__ __forceinline__ float pixel_loss_l4(float dr, float dg, float db)
{
    float dr2 = dr*dr, dg2 = dg*dg, db2 = db*db;
    return dr2*dr2 + dg2*dg2 + db2*db2;
}

/* Binary cross-entropy per channel: -(t*log(p+e) + (1-t)*log(1-p+e)).
 * Takes raw rendered (cr,cg,cb) and target (tr,tg,tb) values in [0,1].
 * Uses fast __logf — acceptable precision for fitness comparison. */
__device__ __forceinline__ float pixel_loss_logll(
    float cr, float cg, float cb,
    float tr, float tg, float tb)
{
    const float eps = 1e-7f;
    return -(tr * __logf(cr + eps) + (1.f - tr) * __logf(1.f - cr + eps))
           -(tg * __logf(cg + eps) + (1.f - tg) * __logf(1.f - cg + eps))
           -(tb * __logf(cb + eps) + (1.f - tb) * __logf(1.f - cb + eps));
}

/* ════════════════════════════════════════════════════════════════════════
 *  Kernel 2 — batch render + inline loss  (GA fitness evaluations)
 *
 *  Templated on LOSS_TYPE (LOSS_MSE or LOSS_L4) so the compiler picks
 *  the right pixel_loss helper at instantiation time — zero runtime branch.
 *
 *  Grid : (ceil(W/RENDER_BLK), ceil(H/RENDER_BLK), count)   Block: (RENDER_BLK, RENDER_BLK, 1)
 *  blockIdx.z = chromosome index.
 *
 *  Each thread renders its pixel for chromosome blockIdx.z, computes
 *  the per-pixel loss vs target, then the block tree-reduces 1024 values
 *  to 1 and atomicAdds into losses[blockIdx.z].
 *
 *  No canvas download — only losses[] (count floats) is transferred back.
 *
 *  Shared memory layout (per block, ~12 KB at RENDER_BLK=32):
 *    sv       [N_VERTEX_GENES]        — triangle vertices
 *    sc       [N_COLOR_GENES]         — triangle colors
 *    s_reduce [RENDER_BLK*RENDER_BLK] — block reduction scratch
 * ════════════════════════════════════════════════════════════════════════ */
template<int LOSS_TYPE>
__global__ void batch_render_loss_kernel(
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

    float pixel_err = 0.f;

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

        int idx=(py*w+px)*3;
        float dr=cr-target[idx], dg=cg-target[idx+1], db=cb-target[idx+2];

        if (LOSS_TYPE == LOSS_L4)
            pixel_err = pixel_loss_l4(dr, dg, db);
        else if (LOSS_TYPE == LOSS_LOGLL)
            pixel_err = pixel_loss_logll(cr, cg, cb,
                                         target[idx], target[idx+1], target[idx+2]);
        else
            pixel_err = pixel_loss_mse(dr, dg, db);
    }

    /* Tree reduction: 256 → 1 within the block */
    s_reduce[tid] = pixel_err;
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
 *  Kernel 3 — batch render + per-channel RGB patch SSIM
 *
 *  WHY per-channel, not luminance:
 *  Luminance-only SSIM (Y = 0.299R+0.587G+0.114B) gives the GA zero
 *  signal about individual colors — any RGB combination with the right
 *  brightness scores identically.  In practice this makes the GA converge
 *  to images with correct brightness but totally wrong hues, which look
 *  like a colour negative of the target.  Computing SSIM independently for
 *  R, G, and B constrains every channel and removes the degeneracy.
 *
 *  Formula (Wang et al. 2004, applied per channel c ∈ {R,G,B}):
 *    SSIM_c = [(2μr_c·μt_c + C1)(2σr_c·t_c + C2)]
 *           / [(μr_c²+μt_c²+C1)(σr_c²+σt_c²+C2)]
 *    loss   = 1 − mean(SSIM_R, SSIM_G, SSIM_B)
 *    C1 = (0.01)² = 1e-4,  C2 = (0.03)² = 9e-4
 *
 *  Each SSIM_BLK×SSIM_BLK (16×16) block is one patch.  For each channel
 *  5 statistics are tree-reduced (Σval_r, Σval_t, Σval_r², Σval_t², Σval_r·val_t)
 *  plus a valid-pixel count — 16 arrays of SSIM_BLK² floats total.
 *  Thread 0 computes the three SSIMs, averages them, and accumulates
 *  (1 − SSIM_avg) × N_valid into losses[chrom].
 *  Host divides by w×h.
 *
 *  Block kept at 16×16 (not 32×32): see SSIM_BLK comment above for rationale.
 *
 *  Shared memory layout (per block, ~24 KB at SSIM_BLK=16):
 *    sv          [N_VERTEX_GENES]           — triangle vertices
 *    sc          [N_COLOR_GENES]            — triangle colors
 *    s_stats[k]  [SSIM_BLK*SSIM_BLK] k=0..15 — reduction scratch
 *      k = ch*5+0  Σ rendered channel ch
 *      k = ch*5+1  Σ target    channel ch
 *      k = ch*5+2  Σ rendered² channel ch
 *      k = ch*5+3  Σ target²   channel ch
 *      k = ch*5+4  Σ rendered×target channel ch
 *      k = 15      valid-pixel count
 *    (ch=0→R, ch=1→G, ch=2→B)
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void batch_render_ssim_kernel(
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
    float *sv      = sdata;
    float *sc      = sv + n_vertex_genes;
    /* 16 reduction arrays each of length block_size, packed after sc */
    float *s_stats = sc + n_color_genes;   /* s_stats[k*block_size + tid] */

    int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x  * blockDim.y;   /* SSIM_BLK * SSIM_BLK */

    for (int i = tid; i < n_vertex_genes; i += block_size) sv[i] = verts[i];
    for (int i = tid; i < n_color_genes;  i += block_size) sc[i] = colors[i];
    __syncthreads();

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    /* Zero all 16 stat slots for this thread */
    for (int k = 0; k < 16; k++) s_stats[k * block_size + tid] = 0.f;

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

        int idx = (py*w + px) * 3;
        float rv[3] = { cr,             cg,             cb             };
        float tv[3] = { target[idx],    target[idx+1],  target[idx+2]  };

        for (int ch = 0; ch < 3; ch++) {
            float r = rv[ch], t = tv[ch];
            s_stats[(ch*5 + 0) * block_size + tid] = r;
            s_stats[(ch*5 + 1) * block_size + tid] = t;
            s_stats[(ch*5 + 2) * block_size + tid] = r * r;
            s_stats[(ch*5 + 3) * block_size + tid] = t * t;
            s_stats[(ch*5 + 4) * block_size + tid] = r * t;
        }
        s_stats[15 * block_size + tid] = 1.f;  /* valid pixel */
    }
    __syncthreads();

    /* 16 parallel tree reductions */
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            for (int k = 0; k < 16; k++)
                s_stats[k * block_size + tid] += s_stats[k * block_size + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float N = s_stats[15 * block_size];
        if (N < 1.f) return;

        const float C1 = 1e-4f;
        const float C2 = 9e-4f;
        float ssim_sum = 0.f;

        for (int ch = 0; ch < 3; ch++) {
            float mu_r  = s_stats[(ch*5+0)*block_size] / N;
            float mu_t  = s_stats[(ch*5+1)*block_size] / N;
            float var_r = s_stats[(ch*5+2)*block_size] / N - mu_r * mu_r;
            float var_t = s_stats[(ch*5+3)*block_size] / N - mu_t * mu_t;
            float cov   = s_stats[(ch*5+4)*block_size] / N - mu_r * mu_t;
            ssim_sum += ((2.f*mu_r*mu_t + C1) * (2.f*cov + C2))
                      / ((mu_r*mu_r + mu_t*mu_t + C1) * (var_r + var_t + C2));
        }

        /* Accumulate pixel-weighted (1 − mean_RGB_SSIM); host normalises by w*h */
        atomicAdd(&losses[chrom], (1.f - ssim_sum / 3.f) * N);
    }
}


/* ════════════════════════════════════════════════════════════════════════
 *  Kernel 4 — batch render + area-weighted MSE  (LOSS_WMSE)
 *
 *  Renders each chromosome and accumulates the per-pixel MSE sum exactly
 *  like the MSE/L4 template.  Thread 0 of every block additionally reads
 *  the chromosome's triangle vertices from shared memory to compute the
 *  total normalised triangle area, then scales the block's partial MSE
 *  sum before the atomicAdd:
 *
 *    weight      = (total_area_in_[0,1]^2_space + 1e-7)^wmse_power
 *    contribution = block_mse_partial_sum * weight
 *
 *  After all blocks: losses[chrom] = weight * total_mse_sum.
 *  Host normalises by w*h*3 giving:  loss = MSE * weight.
 *
 *  Because every block loads the same chromosome's vertices, all
 *  thread-0s compute the identical weight — the decomposition is exact.
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void batch_render_wmse_kernel(
    const float * __restrict__ all_verts,
    const float * __restrict__ all_colors,
    const float * __restrict__ target,
    float       *              losses,
    int w, int h, int n_triangles, int n_vertex_genes, int n_color_genes,
    float wmse_power)
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

    float pixel_err = 0.f;

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

        int idx=(py*w+px)*3;
        float dr=cr-target[idx], dg=cg-target[idx+1], db=cb-target[idx+2];
        pixel_err = dr*dr + dg*dg + db*db;
    }

    /* Tree reduction of per-pixel MSE contributions */
    s_reduce[tid] = pixel_err;
    __syncthreads();
    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        /* Compute total triangle area in normalised [0,1]^2 space.
         * sv[] is still valid here — no code has overwritten shared mem. */
        float norm_area = 0.f;
        for (int t = 0; t < n_triangles; t++) {
            float ax = sv[t*6+2] - sv[t*6+0];
            float ay = sv[t*6+3] - sv[t*6+1];
            float bx = sv[t*6+4] - sv[t*6+0];
            float by = sv[t*6+5] - sv[t*6+1];
            norm_area += fabsf(ax*by - ay*bx) * 0.5f;
        }
        float weight = 1.f + powf(norm_area, wmse_power);
        atomicAdd(&losses[chrom], s_reduce[0] * weight);
    }
}


/* ════════════════════════════════════════════════════════════════════════
 *  Public API
 * ════════════════════════════════════════════════════════════════════════ */

extern "C" int cuda_num_gpus(void) { return g_num_gpus; }

/* cuda_renderer_init
 *
 * Selects GPUs (up to cfg->num_gpus, or all available if 0), uploads the
 * target image to every GPU, and allocates all persistent device + pinned
 * host buffers.  Intended to be called once before the GA loop.
 */
extern "C" void cuda_renderer_init(const Image *target, const AppConfig *cfg)
{
    g_pop_size       = cfg->pop_size;
    g_n_triangles    = cfg->n_triangles;
    g_n_vertex_genes = cfg->n_vertex_genes;
    g_n_color_genes  = cfg->n_color_genes;
    g_n_genes        = cfg->n_genes;

    int w = target->w, h = target->h;

    /* ── Select GPUs ──────────────────────────────────────────────────── */
    int n_avail = 0;
    {
        cudaError_t _e = cudaGetDeviceCount(&n_avail);
        if (_e == cudaErrorInitializationError) {
            fprintf(stderr,
                "[CUDA] Driver initialization failed (cudaErrorInitializationError).\n"
                "       This usually means you are not running on a GPU node.\n"
                "       On AiMOS/Slurm: make sure your job requests a GPU,\n"
                "       e.g.  #SBATCH --gres=gpu:1\n"
                "       Verify with: nvidia-smi\n");
            exit(1);
        }
        if (_e != cudaSuccess) {
            fprintf(stderr, "[CUDA] cudaGetDeviceCount failed: %s\n",
                    cudaGetErrorString(_e));
            exit(1);
        }
    }
    if (n_avail == 0) { fprintf(stderr, "[CUDA] No CUDA devices found.\n"); exit(1); }

    int requested = cfg->num_gpus;
    g_num_gpus = (requested > 0 && requested <= n_avail) ? requested : n_avail;
    if (g_num_gpus > MAX_GPUS) g_num_gpus = MAX_GPUS;

    printf("[CUDA] GPUs available: %d   using: %d\n", n_avail, g_num_gpus);

    /* ── Shared pinned input staging (host → every GPU) ──────────────── */
    CUDA_CHECK(cudaMallocHost(&h_all_verts,
               (size_t)g_pop_size * g_n_vertex_genes * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_all_colors,
               (size_t)g_pop_size * g_n_color_genes  * sizeof(float)));

    /* ── Per-GPU initialisation ───────────────────────────────────────── */
    for (int g = 0; g < g_num_gpus; g++) {
        GpuContext *ctx = &g_gpus[g];
        ctx->device_id  = g;
        ctx->d_verts = ctx->d_colors = ctx->d_canvas = NULL;

        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreate(&ctx->stream));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, g));
        printf("[CUDA] GPU %d : %s  (compute %d.%d, %d SMs)\n",
               g, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

        /* Target image replica — uploaded asynchronously */
        CUDA_CHECK(cudaMalloc(&ctx->d_target, (size_t)w * h * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(ctx->d_target, target->data,
                                   (size_t)w * h * 3 * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));

        /* Batch buffers — sized for the full pop so any slice fits */
        CUDA_CHECK(cudaMalloc(&ctx->d_all_verts,
                   (size_t)g_pop_size * g_n_vertex_genes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->d_all_colors,
                   (size_t)g_pop_size * g_n_color_genes  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx->d_losses,
                   (size_t)g_pop_size                    * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ctx->h_losses,
                   (size_t)g_pop_size                    * sizeof(float)));

        /* GPU 0 also handles single-chromosome PPM snapshot renders */
        if (g == 0) {
            CUDA_CHECK(cudaMalloc(&ctx->d_verts,
                       (size_t)g_n_vertex_genes * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&ctx->d_colors,
                       (size_t)g_n_color_genes  * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&ctx->d_canvas,
                       (size_t)w * h * 3        * sizeof(float)));
        }
    }

    /* Wait for all target uploads before the caller uses any GPU */
    for (int g = 0; g < g_num_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(g_gpus[g].stream));
    }

    printf("[CUDA] Config  : %d triangles  pop=%d  genes=%d\n",
           g_n_triangles, g_pop_size, g_n_genes);
    printf("[CUDA] Buffers : %.1f KB target/GPU  %.1f KB batch verts  %.1f KB batch colors\n",
           w*h*3*sizeof(float)/1024.f,
           (float)(g_pop_size*g_n_vertex_genes*sizeof(float))/1024.f,
           (float)(g_pop_size*g_n_color_genes *sizeof(float))/1024.f);
}

extern "C" void cuda_renderer_free(void)
{
    for (int g = 0; g < g_num_gpus; g++) {
        GpuContext *ctx = &g_gpus[g];
        CUDA_CHECK(cudaSetDevice(ctx->device_id));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaStreamDestroy(ctx->stream));

        cudaFree(ctx->d_target);     ctx->d_target    = NULL;
        cudaFree(ctx->d_all_verts);  ctx->d_all_verts  = NULL;
        cudaFree(ctx->d_all_colors); ctx->d_all_colors = NULL;
        cudaFree(ctx->d_losses);     ctx->d_losses     = NULL;
        cudaFreeHost(ctx->h_losses); ctx->h_losses     = NULL;

        /* GPU 0 single-render buffers (NULL for other GPUs → no-op) */
        cudaFree(ctx->d_verts);  ctx->d_verts  = NULL;
        cudaFree(ctx->d_colors); ctx->d_colors = NULL;
        cudaFree(ctx->d_canvas); ctx->d_canvas = NULL;
    }

    cudaFreeHost(h_all_verts);  h_all_verts  = NULL;
    cudaFreeHost(h_all_colors); h_all_colors = NULL;
    g_num_gpus = 0;
}

/* render_triangles — single render + canvas download, GPU 0 only, PPM snapshots */
extern "C" void render_triangles(const float *vertices_flat, const float *colors,
                                 Image *dst, Profiler *prof)
{
    double t0 = profiler_now();
    GpuContext *ctx = &g_gpus[0];
    int w = dst->w, h = dst->h;

    CUDA_CHECK(cudaSetDevice(ctx->device_id));
    CUDA_CHECK(cudaMemcpy(ctx->d_verts,  vertices_flat,
                          g_n_vertex_genes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_colors, colors,
                          g_n_color_genes  * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(RENDER_BLK, RENDER_BLK);
    dim3 grid((w+RENDER_BLK-1)/RENDER_BLK, (h+RENDER_BLK-1)/RENDER_BLK);
    size_t smem = (size_t)(g_n_vertex_genes + g_n_color_genes) * sizeof(float);

    render_kernel<<<grid, block, smem>>>(ctx->d_verts, ctx->d_colors, ctx->d_canvas, w, h,
                                         g_n_triangles, g_n_vertex_genes, g_n_color_genes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst->data, ctx->d_canvas,
                          w*h*3*sizeof(float), cudaMemcpyDeviceToHost));

    profiler_add(prof, BUCKET_RENDER, profiler_now() - t0);
}

/*
 * batch_compute_loss_gpu  —  multi-GPU fitness evaluation
 *
 * Splits `count` chromosomes evenly across g_num_gpus GPUs.  For each GPU:
 *   1. DMA the chromosome slice H→D (from the shared pinned staging buffer)
 *   2. Zero the loss buffer and launch the render+loss kernel
 *   3. DMA the loss scalars D→H into the per-GPU pinned output buffer
 * Steps 1-3 are queued on each GPU's dedicated stream and run concurrently.
 * A second pass synchronises each stream and assembles losses_out[].
 *
 * Transfer cost per call (per GPU):
 *   Upload  : slice * (N_VERTEX_GENES + N_COLOR_GENES) * 4 bytes
 *   Download: slice * 4 bytes
 */
extern "C" void batch_compute_loss_gpu(const double *pop, int count,
                                        double *losses_out, int w, int h,
                                        Profiler *prof, int loss_type,
                                        double wmse_power)
{
    double t0 = profiler_now();

    /* ── Distribute chromosomes across GPUs ───────────────────────────── */
    int base = count / g_num_gpus;
    int rem  = count % g_num_gpus;

    int starts[MAX_GPUS], slices[MAX_GPUS];
    for (int g = 0, off = 0; g < g_num_gpus; g++) {
        slices[g] = base + (g < rem ? 1 : 0);
        starts[g] = off;
        off += slices[g];
    }

    /* ── Stage all chromosomes into the shared pinned input buffers ───── */
    for (int c = 0; c < count; c++) {
        const double *x      = pop          + (size_t)c * g_n_genes;
        float        *verts  = h_all_verts  + c * g_n_vertex_genes;
        float        *colors = h_all_colors + c * g_n_color_genes;
        for (int i = 0; i < g_n_vertex_genes; i++) verts[i]  = (float)x[i];
        for (int i = 0; i < g_n_color_genes;  i++) colors[i] = (float)x[g_n_vertex_genes + i];
    }

    /* ── Shared memory sizes ──────────────────────────────────────────── */
    size_t smem_pixloss = (size_t)(g_n_vertex_genes + g_n_color_genes
                          + RENDER_BLK * RENDER_BLK) * sizeof(float);
    size_t smem_ssim    = (size_t)(g_n_vertex_genes + g_n_color_genes
                          + 16 * SSIM_BLK * SSIM_BLK) * sizeof(float);

    dim3 block_loss(RENDER_BLK, RENDER_BLK, 1);
    dim3 block_ssim(SSIM_BLK,   SSIM_BLK,   1);

    /* ── Phase 1: launch all GPUs concurrently ────────────────────────── */
    for (int g = 0; g < g_num_gpus; g++) {
        int n = slices[g];
        if (n == 0) continue;
        GpuContext *ctx = &g_gpus[g];
        CUDA_CHECK(cudaSetDevice(ctx->device_id));

        /* H→D: upload this GPU's chromosome slice from the shared staging buffer */
        CUDA_CHECK(cudaMemcpyAsync(
            ctx->d_all_verts,
            h_all_verts  + (size_t)starts[g] * g_n_vertex_genes,
            (size_t)n * g_n_vertex_genes * sizeof(float),
            cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            ctx->d_all_colors,
            h_all_colors + (size_t)starts[g] * g_n_color_genes,
            (size_t)n * g_n_color_genes * sizeof(float),
            cudaMemcpyHostToDevice, ctx->stream));

        /* Zero the loss accumulator before the kernel's atomicAdds */
        CUDA_CHECK(cudaMemsetAsync(ctx->d_losses, 0,
                   (size_t)n * sizeof(float), ctx->stream));

        /* Launch render+loss kernel on this GPU's stream */
        if (loss_type == LOSS_SSIM) {
            dim3 grid((w+SSIM_BLK-1)/SSIM_BLK, (h+SSIM_BLK-1)/SSIM_BLK, n);
            batch_render_ssim_kernel<<<grid, block_ssim, smem_ssim, ctx->stream>>>(
                ctx->d_all_verts, ctx->d_all_colors, ctx->d_target, ctx->d_losses,
                w, h, g_n_triangles, g_n_vertex_genes, g_n_color_genes);
        } else if (loss_type == LOSS_WMSE) {
            dim3 grid((w+RENDER_BLK-1)/RENDER_BLK, (h+RENDER_BLK-1)/RENDER_BLK, n);
            batch_render_wmse_kernel<<<grid, block_loss, smem_pixloss, ctx->stream>>>(
                ctx->d_all_verts, ctx->d_all_colors, ctx->d_target, ctx->d_losses,
                w, h, g_n_triangles, g_n_vertex_genes, g_n_color_genes,
                (float)wmse_power);
        } else {
            dim3 grid((w+RENDER_BLK-1)/RENDER_BLK, (h+RENDER_BLK-1)/RENDER_BLK, n);
            if (loss_type == LOSS_L4)
                batch_render_loss_kernel<LOSS_L4><<<grid, block_loss, smem_pixloss, ctx->stream>>>(
                    ctx->d_all_verts, ctx->d_all_colors, ctx->d_target, ctx->d_losses,
                    w, h, g_n_triangles, g_n_vertex_genes, g_n_color_genes);
            else if (loss_type == LOSS_LOGLL)
                batch_render_loss_kernel<LOSS_LOGLL><<<grid, block_loss, smem_pixloss, ctx->stream>>>(
                    ctx->d_all_verts, ctx->d_all_colors, ctx->d_target, ctx->d_losses,
                    w, h, g_n_triangles, g_n_vertex_genes, g_n_color_genes);
            else
                batch_render_loss_kernel<LOSS_MSE><<<grid, block_loss, smem_pixloss, ctx->stream>>>(
                    ctx->d_all_verts, ctx->d_all_colors, ctx->d_target, ctx->d_losses,
                    w, h, g_n_triangles, g_n_vertex_genes, g_n_color_genes);
        }

        CUDA_CHECK(cudaGetLastError());

        /* D→H: async download into this GPU's own pinned output buffer */
        CUDA_CHECK(cudaMemcpyAsync(ctx->h_losses, ctx->d_losses,
                   (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost, ctx->stream));
    }

    /* ── Phase 2: sync each GPU and assemble losses_out[] ────────────── */
    /* SSIM accumulates pixel-weighted values normalised by pixel count;
     * all other losses accumulate per-channel sums normalised by w*h*3. */
    double norm = (loss_type == LOSS_SSIM) ? (double)w * h
                                           : (double)w * h * 3;
    for (int g = 0; g < g_num_gpus; g++) {
        int n = slices[g];
        if (n == 0) continue;
        GpuContext *ctx = &g_gpus[g];
        CUDA_CHECK(cudaSetDevice(ctx->device_id));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        for (int c = 0; c < n; c++)
            losses_out[starts[g] + c] = (double)ctx->h_losses[c] / norm;
    }

    if (prof) profiler_add(prof, BUCKET_RENDER, profiler_now() - t0);
}

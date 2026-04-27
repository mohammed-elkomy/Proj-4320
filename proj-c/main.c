/*  main.c  —  Simulation driver
 *
 *  Loads target.ppm, runs the GA to approximate it with triangles,
 *  saves PPM snapshots, and prints the timing report.
 *
 *  Output files:
 *      progress_GA_genNNNN.ppm    — snapshot every VISUALISE_EVERY generations
 *      final_result.ppm           — best result at termination
 *      run.log                    — copy of all console output
 */

#include "triangle_ga.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#define TARGET_FILE     "target.ppm"
#define LOG_FILE        "run.log"
#define SAVE_FILE       "ga_checkpoint.bin"
#define PROGRESS_DIR    "progress_ppm"
#define VISUALISE_EVERY 10000

/* Mirror every printf to both stdout and the log file */
static FILE *g_log = NULL;

static void logprintf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if (g_log) {
        va_start(args, fmt);
        vfprintf(g_log, fmt, args);
        va_end(args);
        fflush(g_log);
    }
}

/* ── CPU renderer (mirror of render.c, used only for benchmarking) ──────── */
static void render_triangles_cpu(const float *vertices_flat, const float *colors,
                                 Image *dst)
{
    const int w = dst->w, h = dst->h;
    float *canvas = dst->data;

    for (int i = 0; i < w * h * 3; i++) canvas[i] = 1.0f;

    for (int t = 0; t < N_TRIANGLES; t++) {
        const float *vb = vertices_flat + t * 6;
        float x0 = vb[0]*w, y0 = vb[1]*h;
        float x1 = vb[2]*w, y1 = vb[3]*h;
        float x2 = vb[4]*w, y2 = vb[5]*h;
        float r = colors[t*4+0], g = colors[t*4+1];
        float b = colors[t*4+2], a = colors[t*4+3];

        float v0x = x2-x0, v0y = y2-y0;
        float v1x = x1-x0, v1y = y1-y0;
        float d00 = v0x*v0x+v0y*v0y, d01 = v0x*v1x+v0y*v1y;
        float d11 = v1x*v1x+v1y*v1y;
        float denom = d00*d11 - d01*d01;
        if (fabsf(denom) < 1e-10f) continue;
        float inv = 1.0f / denom;

        int ix0 = (int)fminf(fminf(x0,x1),x2);   if (ix0 < 0) ix0 = 0;
        int iy0 = (int)fminf(fminf(y0,y1),y2);   if (iy0 < 0) iy0 = 0;
        int ix1 = (int)fmaxf(fmaxf(x0,x1),x2)+1; if (ix1 > w) ix1 = w;
        int iy1 = (int)fmaxf(fmaxf(y0,y1),y2)+1; if (iy1 > h) iy1 = h;

        float one_minus_a = 1.0f - a;
        for (int py = iy0; py < iy1; py++) {
            float d02y = v0y*(py-y0), d12y = v1y*(py-y0);
            for (int px = ix0; px < ix1; px++) {
                float v2x = px-x0;
                float d02 = v0x*v2x + d02y;
                float d12 = v1x*v2x + d12y;
                float u = (d11*d02 - d01*d12) * inv;
                float v = (d00*d12 - d01*d02) * inv;
                if (u >= 0.0f && v >= 0.0f && u+v <= 1.0f) {
                    int idx = (py*w+px)*3;
                    canvas[idx+0] = r*a + canvas[idx+0]*one_minus_a;
                    canvas[idx+1] = g*a + canvas[idx+1]*one_minus_a;
                    canvas[idx+2] = b*a + canvas[idx+2]*one_minus_a;
                }
            }
        }
    }
}

/* ── Render benchmark: times CPU vs GPU over N_BENCH iterations ─────────── */
#define N_BENCH 50

static void benchmark_rendering(const Image *target, Profiler *prof)
{
    /* ── Build one random chromosome for the single-render tests ── */
    float verts[N_VERTEX_GENES], cols[N_COLOR_GENES];
    double x[N_GENES];
    for (int i = 0; i < N_GENES; i++) x[i] = rng_uniform();
    vec_to_parts(x, verts, cols);

    Image *scratch = image_alloc(target->w, target->h);

    /* ── Section 1: single render  CPU vs GPU ── */
    double cpu_start = profiler_now();
    for (int i = 0; i < N_BENCH; i++)
        render_triangles_cpu(verts, cols, scratch);
    double cpu_total = profiler_now() - cpu_start;
    double cpu_mean  = cpu_total / N_BENCH * 1000.0;

    render_triangles(verts, cols, scratch, prof);   /* warm-up */
    prof->totals[BUCKET_RENDER] = 0.0;
    prof->counts[BUCKET_RENDER] = 0;

    double gpu_start = profiler_now();
    for (int i = 0; i < N_BENCH; i++)
        render_triangles(verts, cols, scratch, prof);
    double gpu_total = profiler_now() - gpu_start;
    double gpu_mean  = gpu_total / N_BENCH * 1000.0;

    prof->totals[BUCKET_RENDER] = 0.0;
    prof->counts[BUCKET_RENDER] = 0;

    /* ── Section 2: full-population fitness  sequential vs batch ── */

    /* Build a fake population of POP_SIZE random chromosomes */
    static double fake_pop[POP_SIZE][N_GENES];
    for (int c = 0; c < POP_SIZE; c++)
        for (int j = 0; j < N_GENES; j++)
            fake_pop[c][j] = rng_uniform();

    double seq_losses[POP_SIZE];
    double batch_losses[POP_SIZE];

    /* Sequential: POP_SIZE × (render_triangles + CPU MSE) — old path */
    double seq_start = profiler_now();
    for (int r = 0; r < N_BENCH; r++) {
        for (int c = 0; c < POP_SIZE; c++) {
            float fv[N_VERTEX_GENES], fc[N_COLOR_GENES];
            vec_to_parts(fake_pop[c], fv, fc);
            render_triangles(fv, fc, scratch, prof);   /* GPU render + canvas download */
            int n = target->w * target->h * 3;
            double mse = 0.0;
            for (int k = 0; k < n; k++) {
                double d = (double)scratch->data[k] - (double)target->data[k];
                mse += d * d;
            }
            seq_losses[c] = mse / n;
        }
    }
    double seq_total = profiler_now() - seq_start;
    double seq_mean  = seq_total / N_BENCH * 1000.0;
    prof->totals[BUCKET_RENDER] = 0.0;
    prof->counts[BUCKET_RENDER] = 0;
    (void)seq_losses;

    /* Batch: one batch_compute_loss_gpu call for all POP_SIZE chromosomes */
    batch_compute_loss_gpu((const double *)fake_pop, POP_SIZE,
                           batch_losses, target->w, target->h);   /* warm-up */

    double bat_start = profiler_now();
    for (int r = 0; r < N_BENCH; r++)
        batch_compute_loss_gpu((const double *)fake_pop, POP_SIZE,
                               batch_losses, target->w, target->h);
    double bat_total = profiler_now() - bat_start;
    double bat_mean  = bat_total / N_BENCH * 1000.0;
    (void)batch_losses;

    image_free(scratch);

    printf("\n=== Render Benchmark (%d iterations, %dx%d image) ===\n",
           N_BENCH, target->w, target->h);
    printf("\n  [Single render]\n");
    printf("    CPU render     : %8.3f ms/call\n", cpu_mean);
    printf("    GPU render     : %8.3f ms/call\n", gpu_mean);
    printf("    Speedup        : %.2fx\n", cpu_mean / gpu_mean);
    printf("\n  [Full-population fitness (%d chromosomes)]\n", POP_SIZE);
    printf("    Sequential GPU : %8.3f ms/gen  (%d renders + CPU MSE)\n",
           seq_mean, POP_SIZE);
    printf("    Batch GPU      : %8.3f ms/gen  (1 kernel, MSE on GPU)\n",
           bat_mean);
    printf("    Speedup        : %.2fx\n", seq_mean / bat_mean);
    printf("=====================================================\n\n");
}

/* ── Save a side-by-side comparison (target | current best) as PPM ──────── */
static void save_sidebyside(const Image *target, const double *x,
                            const char *path, Profiler *prof)
{
    float verts[N_VERTEX_GENES], cols[N_COLOR_GENES];
    vec_to_parts(x, verts, cols);

    Image *rendered = image_alloc(target->w, target->h);
    render_triangles(verts, cols, rendered, prof);

    Image *combined = image_alloc(target->w * 2, target->h);
    for (int y = 0; y < target->h; y++) {
        memcpy(combined->data + (y * target->w * 2) * 3,
               target->data   + (y * target->w) * 3,
               target->w * 3 * sizeof(float));
        memcpy(combined->data + (y * target->w * 2 + target->w) * 3,
               rendered->data + (y * target->w) * 3,
               target->w * 3 * sizeof(float));
    }

    image_write_ppm(combined, path);
    image_free(rendered);
    image_free(combined);
}

/* ══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    g_log = fopen(LOG_FILE, "w");
    mkdir(PROGRESS_DIR, 0755);

    Profiler prof;
    profiler_init(&prof);

    rng_seed(42);

    /* Load target image */
    Image *target = image_read_ppm(TARGET_FILE);
    if (!target) {
        fprintf(stderr, "Error: could not load '%s'. "
                        "Place a P6 PPM file named target.ppm in this directory.\n",
                TARGET_FILE);
        return 1;
    }

    logprintf("Target image : %d x %d (loaded from %s)\n",
              target->w, target->h, TARGET_FILE);
    logprintf("State vector : length %d  "
              "(%d triangles x (3 vertices x 2 coords + 4 RGBA))\n",
              N_GENES, N_TRIANGLES);
    logprintf("Solver       : GA\n\n");

    /* Allocate persistent GPU buffers sized to this image */
    cuda_renderer_init(target);
    benchmark_rendering(target, &prof);

    /* Run GA — resume from checkpoint if one exists */
    GA ga;
    ga_init(&ga, target, &prof);
    if (ga_load(&ga, SAVE_FILE) == 0)
        logprintf("Resumed from %s at generation %d  (best=%.6f)\n\n",
                  SAVE_FILE, ga.generation, ga.best_loss);
    else
        logprintf("No checkpoint found — starting fresh.\n\n");

    while (!ga_is_done(&ga)) {
        GAStats st = ga_step(&ga);

        printf("Gen %4d | best=%.6f | avg=%.6f | stagnation=%d/%d\n",
               st.generation, st.best_loss, st.avg_loss,
               ga.stagnation_count, STAGNATION_GENS);

        if (ga.stagnation_count == 0 && g_log)
            fprintf(g_log, "Gen %4d | best=%.6f\n",
                    st.generation, st.best_loss);

        if (st.generation % VISUALISE_EVERY == 0 || ga_is_done(&ga)) {
            char path[128];
            snprintf(path, sizeof(path),
                     PROGRESS_DIR "/progress_GA_gen%06d.ppm", st.generation);
            save_sidebyside(target, ga.population[st.best_idx], path, &prof);
            ga_save(&ga, SAVE_FILE);
        }
    }

    /* Final output */
    GAStats final_st = ga_stats(&ga);
    save_sidebyside(target, ga.population[final_st.best_idx],
                    "final_result.ppm", &prof);

    logprintf("\nDone. Final loss=%.6f  Saved final_result.ppm\n",
              final_st.best_loss);

    /* Fix up profiler: optimize bucket should exclude render time */
    prof.totals[BUCKET_OPTIMIZE] -= prof.totals[BUCKET_RENDER];

    profiler_report(&prof);

    ga_save(&ga, SAVE_FILE);
    ga_destroy(&ga);
    cuda_renderer_free();
    image_free(target);

    if (g_log) fclose(g_log);
    return 0;
}

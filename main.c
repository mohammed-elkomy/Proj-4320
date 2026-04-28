/*  main.c  —  Simulation driver
 *
 *  Loads input/target.ppm, runs the GA to approximate it with triangles,
 *  saves PPM snapshots, and prints the timing report.
 *
 *  Input:
 *      input/target.ppm
 *
 *  Output (under output/RUN_PREFIX_YYYY-MM-DD_HH-MM-SS/):
 *      progress_ppm/progress_GA_genNNNNNN.ppm  — snapshots
 *      final_result.ppm                        — best result
 *      run.log                                 — console log
 *      ga_checkpoint.bin                       — resume state
 */

#include "app.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#define TARGET_FILE  "input/target.ppm"
#define CONFIG_DIR   "config"
#define DEFAULT_CONF "config/default.conf"

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
                                 Image *dst, const AppConfig *cfg)
{
    const int w = dst->w, h = dst->h;
    float *canvas = dst->data;

    for (int i = 0; i < w * h * 3; i++) canvas[i] = 1.0f;

    for (int t = 0; t < cfg->n_triangles; t++) {
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

static void benchmark_rendering(const Image *target, Profiler *prof,
                                const AppConfig *cfg)
{
    int n_genes        = cfg->n_genes;
    int n_vertex_genes = cfg->n_vertex_genes;
    int n_color_genes  = cfg->n_color_genes;
    int pop_size       = cfg->pop_size;

    float  *verts = malloc(n_vertex_genes * sizeof(float));
    float  *cols  = malloc(n_color_genes  * sizeof(float));
    double *x     = malloc(n_genes        * sizeof(double));
    for (int i = 0; i < n_genes; i++) x[i] = rng_uniform();
    vec_to_parts(x, verts, cols, cfg);

    Image *scratch = image_alloc(target->w, target->h);

    double cpu_start = profiler_now();
    for (int i = 0; i < N_BENCH; i++)
        render_triangles_cpu(verts, cols, scratch, cfg);
    double cpu_mean = (profiler_now() - cpu_start) / N_BENCH * 1000.0;

    render_triangles(verts, cols, scratch, prof);
    prof->totals[BUCKET_RENDER] = 0.0;
    prof->counts[BUCKET_RENDER] = 0;

    double gpu_start = profiler_now();
    for (int i = 0; i < N_BENCH; i++)
        render_triangles(verts, cols, scratch, prof);
    double gpu_mean = (profiler_now() - gpu_start) / N_BENCH * 1000.0;
    prof->totals[BUCKET_RENDER] = 0.0;
    prof->counts[BUCKET_RENDER] = 0;

    double *fake_pop   = malloc((size_t)pop_size * n_genes * sizeof(double));
    double *seq_losses = malloc(pop_size * sizeof(double));
    double *bat_losses = malloc(pop_size * sizeof(double));
    for (int c = 0; c < pop_size; c++)
        for (int j = 0; j < n_genes; j++)
            fake_pop[(size_t)c * n_genes + j] = rng_uniform();

    float *fv = malloc(n_vertex_genes * sizeof(float));
    float *fc = malloc(n_color_genes  * sizeof(float));

    int loss_type = cfg->loss_type;
    int npix_rgb = target->w * target->h * 3;
    int npix     = target->w * target->h;

    double seq_start = profiler_now();
    for (int r = 0; r < N_BENCH; r++) {
        for (int c = 0; c < pop_size; c++) {
            vec_to_parts(fake_pop + (size_t)c * n_genes, fv, fc, cfg);
            render_triangles(fv, fc, scratch, prof);

            if (loss_type == LOSS_SSIM) {
                /* Global single-patch SSIM on luminance (timing benchmark) */
                double sr=0, st=0, sr2=0, st2=0, srt=0;
                for (int k = 0; k < npix; k++) {
                    double lr = 0.299*scratch->data[k*3+0] + 0.587*scratch->data[k*3+1]
                              + 0.114*scratch->data[k*3+2];
                    double lt = 0.299*target->data[k*3+0]  + 0.587*target->data[k*3+1]
                              + 0.114*target->data[k*3+2];
                    sr += lr; st += lt; sr2 += lr*lr; st2 += lt*lt; srt += lr*lt;
                }
                double N = (double)npix;
                double mu_r = sr/N, mu_t = st/N;
                double vr = sr2/N - mu_r*mu_r, vt = st2/N - mu_t*mu_t;
                double cov = srt/N - mu_r*mu_t;
                double C1=1e-4, C2=9e-4;
                double ssim = ((2*mu_r*mu_t+C1)*(2*cov+C2))
                            / ((mu_r*mu_r+mu_t*mu_t+C1)*(vr+vt+C2));
                seq_losses[c] = 1.0 - ssim;
            } else if (loss_type == LOSS_L4) {
                double acc = 0.0;
                for (int k = 0; k < npix_rgb; k++) {
                    double d = (double)scratch->data[k] - (double)target->data[k];
                    double d2 = d * d;
                    acc += d2 * d2;
                }
                seq_losses[c] = acc / npix_rgb;
            } else {
                double acc = 0.0;
                for (int k = 0; k < npix_rgb; k++) {
                    double d = (double)scratch->data[k] - (double)target->data[k];
                    acc += d * d;
                }
                seq_losses[c] = acc / npix_rgb;
            }
        }
    }
    double seq_mean = (profiler_now() - seq_start) / N_BENCH * 1000.0;
    prof->totals[BUCKET_RENDER] = 0.0;
    prof->counts[BUCKET_RENDER] = 0;
    (void)seq_losses;

    batch_compute_loss_gpu(fake_pop, pop_size, bat_losses,
                           target->w, target->h, NULL, loss_type);

    double bat_start = profiler_now();
    for (int r = 0; r < N_BENCH; r++)
        batch_compute_loss_gpu(fake_pop, pop_size, bat_losses,
                               target->w, target->h, NULL, loss_type);
    double bat_mean = (profiler_now() - bat_start) / N_BENCH * 1000.0;
    (void)bat_losses;

    free(verts); free(cols); free(x);
    free(fake_pop); free(seq_losses); free(bat_losses);
    free(fv); free(fc);
    image_free(scratch);

    const char *loss_name = (loss_type == LOSS_SSIM) ? "SSIM"
                          : (loss_type == LOSS_L4)  ? "L4" : "MSE";
    printf("\n=== Render Benchmark (%d iterations, %dx%d image, loss=%s) ===\n",
           N_BENCH, target->w, target->h, loss_name);
    printf("\n  [Single render]\n");
    printf("    CPU render     : %8.3f ms/call\n", cpu_mean);
    printf("    GPU render     : %8.3f ms/call\n", gpu_mean);
    printf("    Speedup        : %.2fx\n", cpu_mean / gpu_mean);
    printf("\n  [Full-population fitness (%d chromosomes)]\n", pop_size);
    printf("    Sequential GPU : %8.3f ms/gen  (%d renders + CPU %s)\n",
           seq_mean, pop_size, loss_name);
    printf("    Batch GPU      : %8.3f ms/gen  (1 kernel, %s, %d GPU(s))\n",
           bat_mean, loss_name, cuda_num_gpus());
    printf("    Speedup        : %.2fx\n", seq_mean / bat_mean);
    printf("=====================================================\n\n");
}

/* ── Save a side-by-side comparison (target | current best) as PPM ──────── */
static void save_sidebyside(const Image *target, const double *x,
                            const char *path, Profiler *prof,
                            const AppConfig *cfg)
{
    float *verts = malloc(cfg->n_vertex_genes * sizeof(float));
    float *cols  = malloc(cfg->n_color_genes  * sizeof(float));
    vec_to_parts(x, verts, cols, cfg);

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
    free(verts);
    free(cols);
}

/* ══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    /* Load config — argv[1] overrides the default path */
    const char *conf_path = (argc > 1) ? argv[1] : DEFAULT_CONF;
    AppConfig cfg;
    app_config_init(&cfg);
    mkdir(CONFIG_DIR, 0755);
    if (app_config_load(&cfg, conf_path) == 0)
        printf("Config       : loaded from %s\n", conf_path);
    else
        printf("Config       : %s not found — using built-in defaults\n", conf_path);
    app_config_finalize(&cfg);

    /* Build timestamped output directory: output/PREFIX_YYYY-MM-DD_HH-MM-SS */
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char out_dir[256], progress_dir[512], log_file[512], save_file[512], final_file[512];
    snprintf(out_dir, sizeof(out_dir),
             "output/%s_%04d-%02d-%02d_%02d-%02d-%02d",
             cfg.run_prefix,
             t->tm_year+1900, t->tm_mon+1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
    snprintf(progress_dir, sizeof(progress_dir), "%s/progress_ppm",      out_dir);
    snprintf(log_file,     sizeof(log_file),     "%s/run.log",            out_dir);
    snprintf(save_file,    sizeof(save_file),     "%s/ga_checkpoint.bin", out_dir);
    snprintf(final_file,   sizeof(final_file),    "%s/final_result.ppm",  out_dir);

    mkdir("output",     0755);
    mkdir(out_dir,      0755);
    mkdir(progress_dir, 0755);

    /* Save full hyperparams (compile-time + runtime) into the output folder */
    char params_file[512];
    snprintf(params_file, sizeof(params_file), "%s/hyperparams.txt", out_dir);
    app_config_save(&cfg, params_file);

    g_log = fopen(log_file, "w");

    Profiler prof;
    profiler_init(&prof);

    rng_seed(42);

    /* Load target image */
    Image *target = image_read_ppm(TARGET_FILE);
    if (!target) {
        fprintf(stderr, "Error: could not load '%s'.\n", TARGET_FILE);
        return 1;
    }

    logprintf("Output dir   : %s\n", out_dir);
    logprintf("Target image : %d x %d (loaded from %s)\n",
              target->w, target->h, TARGET_FILE);
    logprintf("State vector : length %d  "
              "(%d triangles x (3 vertices x 2 coords + 4 RGBA))\n",
              cfg.n_genes, cfg.n_triangles);
    logprintf("Solver       : GA\n\n");

    /* Allocate persistent GPU buffers sized to this image */
    cuda_renderer_init(target, &cfg);
    benchmark_rendering(target, &prof, &cfg);

    /* Run GA — resume from checkpoint if one exists */
    double wall_start = profiler_now();
    GA ga;
    ga_init(&ga, target, &prof, &cfg);
    if (ga_load(&ga, save_file) == 0)
        logprintf("Resumed from %s at generation %d  (best=%.6f)\n\n",
                  save_file, ga.generation, ga.best_loss);
    else
        logprintf("No checkpoint found — starting fresh.\n\n");

    while (!ga_is_done(&ga)) {
        GAStats st = ga_step(&ga);

        if (cfg.max_generations > 0) {
            /* Hard generation limit active — stagnation is disabled */
            printf("Gen %6d/%-6d | best=%.6f | avg=%.6f\n",
                   st.generation, cfg.max_generations,
                   st.best_loss, st.avg_loss);
        } else {
            printf("Gen %6d | best=%.6f | avg=%.6f | stag=%d/%d\n",
                   st.generation, st.best_loss, st.avg_loss,
                   ga.stagnation_count,
                   cfg.stagnation_gens > 0 ? cfg.stagnation_gens : -1);
        }

        if (ga.stagnation_count == 0 && g_log)
            fprintf(g_log, "Gen %6d | best=%.6f\n", st.generation, st.best_loss);

        if (st.generation % cfg.visualise_every == 0 || ga_is_done(&ga)) {
            char path[512];
            snprintf(path, sizeof(path),
                     "%s/progress_ppm/progress_GA_gen%06d.ppm", out_dir, st.generation);
            save_sidebyside(target, GA_CHROM(&ga, st.best_idx), path, &prof, &cfg);
            ga_save(&ga, save_file);
        }
    }

    /* Final output */
    GAStats final_st = ga_stats(&ga);
    save_sidebyside(target, GA_CHROM(&ga, final_st.best_idx), final_file, &prof, &cfg);

    double wall_elapsed = profiler_now() - wall_start;
    logprintf("\nDone. Final loss=%.6f  Saved %s\n", final_st.best_loss, final_file);
    logprintf("Wall time    : %.2f s  (%d generations)\n",
              wall_elapsed, final_st.generation);

    /* Fix up profiler: optimize bucket should exclude render time */
    prof.totals[BUCKET_OPTIMIZE] -= prof.totals[BUCKET_RENDER];

    profiler_report(&prof, g_log);

    ga_save(&ga, save_file);
    ga_destroy(&ga);
    cuda_renderer_free();
    image_free(target);

    if (g_log) fclose(g_log);
    return 0;
}

/*  main.c  —  MPI + CUDA GA driver (island model)
 *
 *  Each MPI rank runs an independent GA island, seeded differently so that
 *  islands explore distinct regions of the search space.  Every rank uses
 *  all CUDA GPUs visible on its node for batch fitness evaluation.
 *
 *  Every `migration_interval` generations, each rank sends its best
 *  `migration_size` chromosomes to the next rank in a ring (0→1→…→N−1→0)
 *  and receives the same number from the previous rank.  Received chromosomes
 *  replace the worst slots and are immediately re-evaluated before the next
 *  GA step.
 *
 *  Termination: any island hitting its stopping criterion (stagnation or
 *  max_generations) causes all islands to stop via MPI_Allreduce(MPI_LOR).
 *
 *  File I/O (logs, PPM snapshots, checkpoints, hyperparams) is performed
 *  exclusively by rank 0.  At the end, the globally best chromosome across
 *  all ranks is gathered to rank 0 and saved.
 *
 *  Usage:
 *    mpirun -np <N> ./app.o [config_file]
 */

#include "app.h"
#include <mpi.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#define CONFIG_DIR   "config"
#define DEFAULT_CONF "config/default.conf"

static int   g_world_rank = 0;
static int   g_world_size = 1;
static FILE *g_log        = NULL;

/* ── Logging: stdout + log file, rank 0 only ────────────────────────────── */
static void logprintf(const char *fmt, ...)
{
    if (g_world_rank != 0) return;
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

/* ── CPU renderer (used only by the rendering benchmark on rank 0) ──────── */
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

/* ── Render benchmark: CPU vs GPU, rank 0 only ──────────────────────────── */
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
                           target->w, target->h, NULL, loss_type, 0.5);

    double bat_start = profiler_now();
    for (int r = 0; r < N_BENCH; r++)
        batch_compute_loss_gpu(fake_pop, pop_size, bat_losses,
                               target->w, target->h, NULL, loss_type, 0.5);
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

/* ── Save side-by-side comparison PPM (target | current best) ───────────── */
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

/* ── MPI ring migration ─────────────────────────────────────────────────────
 *
 *  After ga_step, the population is sorted ascending (index 0 = best).
 *  We send the best `mig` chromosomes to the next rank (right) and receive
 *  `mig` chromosomes from the previous rank (left).
 *  Received chromosomes replace the `mig` worst slots and are immediately
 *  re-evaluated so the next ga_step sees correct fitnesses.
 */
static void do_migration(GA *ga, const AppConfig *cfg,
                         int rank, int nprocs, Profiler *prof)
{
    int mig      = cfg->migration_size;
    int n_genes  = cfg->n_genes;
    int pop_size = cfg->pop_size;

    if (mig <= 0 || mig > pop_size / 2) return;

    double *send_buf = malloc((size_t)mig * n_genes * sizeof(double));
    double *recv_buf = malloc((size_t)mig * n_genes * sizeof(double));

    /* Best chromosomes are at index 0 after sorting in ga_step */
    memcpy(send_buf, ga->population, (size_t)mig * n_genes * sizeof(double));

    int right = (rank + 1) % nprocs;
    int left  = (rank - 1 + nprocs) % nprocs;

    MPI_Sendrecv(
        send_buf, (int)(mig * n_genes), MPI_DOUBLE, right, 0,
        recv_buf, (int)(mig * n_genes), MPI_DOUBLE, left,  0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Overwrite the worst mig slots with received immigrants */
    double *worst_slot = ga->population + (size_t)(pop_size - mig) * n_genes;
    memcpy(worst_slot, recv_buf, (size_t)mig * n_genes * sizeof(double));

    /* Re-evaluate immigrants so ga_step can sort them correctly */
    batch_compute_loss_gpu(
        worst_slot, mig,
        ga->fitnesses + (pop_size - mig),
        ga->target->w, ga->target->h,
        prof, cfg->loss_type, cfg->wmse_power);

    free(send_buf);
    free(recv_buf);
}

/* ══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank);

    /* ── Config: rank 0 loads + finalizes, then broadcasts to all ranks ── */
    const char *conf_path = (argc > 1) ? argv[1] : DEFAULT_CONF;
    AppConfig cfg;
    app_config_init(&cfg);

    if (g_world_rank == 0) {
        mkdir(CONFIG_DIR, 0755);
        if (app_config_load(&cfg, conf_path) == 0)
            printf("Config       : loaded from %s\n", conf_path);
        else
            printf("Config       : %s not found — using built-in defaults\n", conf_path);
        app_config_finalize(&cfg);
    }
    /* Broadcast the fully finalised struct to every rank */
    MPI_Bcast(&cfg, sizeof(AppConfig), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ── Distribute generation budget across islands ─────────────────── *
     * Each rank runs max_generations/world_size gens so the total work   *
     * across all nodes equals the configured budget.                      *
     * Stagnation mode (max_generations == 0) is unaffected.              */
    int total_max_generations = cfg.max_generations;
    if (cfg.max_generations > 0 && g_world_size > 1)
        cfg.max_generations = cfg.max_generations / g_world_size;
    if (cfg.visualise_every > 0 && g_world_size > 1)
        cfg.visualise_every = cfg.visualise_every / g_world_size;

    /* ── Output directories and log file (rank 0 only) ────────────────── */
    char out_dir[256], progress_dir[512], log_file[512];
    char save_file[512], final_file[512];

    if (g_world_rank == 0) {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        snprintf(out_dir, sizeof(out_dir),
                 "output/%s_%04d-%02d-%02d_%02d-%02d-%02d",
                 cfg.run_prefix,
                 t->tm_year+1900, t->tm_mon+1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
        snprintf(progress_dir, sizeof(progress_dir), "%s/progress_ppm",      out_dir);
        snprintf(log_file,     sizeof(log_file),     "%s/run.log",            out_dir);
        snprintf(save_file,    sizeof(save_file),    "%s/ga_checkpoint.bin",  out_dir);
        snprintf(final_file,   sizeof(final_file),   "%s/final_result.ppm",   out_dir);

        mkdir("output",     0755);
        mkdir(out_dir,      0755);
        mkdir(progress_dir, 0755);

        char params_file[512];
        snprintf(params_file, sizeof(params_file), "%s/hyperparams.txt", out_dir);
        app_config_save(&cfg, params_file);

        g_log = fopen(log_file, "w");
    }

    /* ── Target image: rank 0 reads, all ranks receive via Bcast ─────── */
    int img_w = 0, img_h = 0;
    Image *target = NULL;

    if (g_world_rank == 0) {
        target = image_read_ppm(cfg.target_file);
        if (!target) {
            fprintf(stderr, "Error: could not load '%s'.\n", cfg.target_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        img_w = target->w;
        img_h = target->h;
    }
    MPI_Bcast(&img_w, 1, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Bcast(&img_h, 1, MPI_INT,   0, MPI_COMM_WORLD);
    if (g_world_rank != 0)
        target = image_alloc(img_w, img_h);
    MPI_Bcast(target->data, img_w * img_h * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ── Profiler and RNG — each rank gets a distinct seed ───────────── */
    Profiler prof;
    profiler_init(&prof);
    rng_seed((uint64_t)(42 + g_world_rank));

    /* ── CUDA init — each rank uses all GPUs visible on its node ────── */
    cuda_renderer_init(target, &cfg);

    /* ── Render benchmark — rank 0 only ─────────────────────────────── */
    if (g_world_rank == 0) {
        logprintf("Output dir   : %s\n", out_dir);
        logprintf("Target image : %d x %d (loaded from %s)\n",
                  img_w, img_h, cfg.target_file);
        logprintf("State vector : length %d  "
                  "(%d triangles x (3 vertices x 2 coords + 4 RGBA))\n",
                  cfg.n_genes, cfg.n_triangles);
        logprintf("MPI ranks    : %d   (island model)\n", g_world_size);
        if (total_max_generations > 0 && g_world_size > 1)
            logprintf("Gen budget   : %d total / %d ranks = %d gens/island\n",
                      total_max_generations, g_world_size, cfg.max_generations);
        if (g_world_size > 1 && cfg.migration_interval > 0)
            logprintf("Migration    : every %d gens, %d chromosome(s) in ring\n",
                      cfg.migration_interval, cfg.migration_size);
        logprintf("Solver       : GA\n\n");

        benchmark_rendering(target, &prof, &cfg);
    }

    /* ── GA loop ─────────────────────────────────────────────────────── */
    double wall_start = profiler_now();
    GA ga;
    ga_init(&ga, target, &prof, &cfg);

    /* Only rank 0 attempts checkpoint resume so islands stay diverse */
    if (g_world_rank == 0 && ga_load(&ga, save_file) == 0)
        logprintf("Resumed from %s at generation %d  (best=%.6f)\n\n",
                  save_file, ga.generation, ga.best_loss);
    else if (g_world_rank == 0)
        logprintf("No checkpoint found — starting fresh.\n\n");

    while (1) {
        /* All ranks must agree on termination before breaking */
        {
            int local_done  = ga_is_done(&ga);
            int global_done = local_done;
            if (g_world_size > 1)
                MPI_Allreduce(&local_done, &global_done, 1,
                              MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            if (global_done) break;
        }

        GAStats st = ga_step(&ga);

        /* Ring migration */
        if (g_world_size > 1 && cfg.migration_interval > 0 &&
            st.generation % cfg.migration_interval == 0)
            do_migration(&ga, &cfg, g_world_rank, g_world_size, &prof);

        /* Progress output — rank 0 only */
        if (g_world_rank == 0) {
            if (cfg.max_generations > 0) {
                printf("[rank 0] Gen %6d/%-6d | best=%.6f | avg=%.6f\n",
                       st.generation, cfg.max_generations,
                       st.best_loss, st.avg_loss);
            } else {
                printf("[rank 0] Gen %6d | best=%.6f | avg=%.6f | stag=%d/%d\n",
                       st.generation, st.best_loss, st.avg_loss,
                       ga.stagnation_count,
                       cfg.stagnation_gens > 0 ? cfg.stagnation_gens : -1);
            }
            if (ga.stagnation_count == 0 && g_log)
                fprintf(g_log, "Gen %6d | best=%.6f\n",
                        st.generation, st.best_loss);
        }

        /* Save snapshot + checkpoint (rank 0, every visualise_every gens) */
        if (g_world_rank == 0 &&
            st.generation % cfg.visualise_every == 0) {
            char path[512];
            snprintf(path, sizeof(path),
                     "%s/progress_ppm/progress_GA_gen%06d.ppm",
                     out_dir, st.generation);
            save_sidebyside(target, GA_CHROM(&ga, st.best_idx), path, &prof, &cfg);
            ga_save(&ga, save_file);
        }
    }

    /* ── Collect global best across all islands ──────────────────────── */
    GAStats final_st = ga_stats(&ga);
    int n_genes = cfg.n_genes;

    /* Gather each rank's best chromosome and its loss to rank 0 */
    double *all_chroms  = NULL;
    double *all_losses  = NULL;
    if (g_world_rank == 0) {
        all_chroms = malloc((size_t)g_world_size * n_genes * sizeof(double));
        all_losses = malloc((size_t)g_world_size * sizeof(double));
    }

    MPI_Gather(GA_CHROM(&ga, final_st.best_idx), n_genes, MPI_DOUBLE,
               all_chroms, n_genes, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    MPI_Gather(&final_st.best_loss, 1, MPI_DOUBLE,
               all_losses, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (g_world_rank == 0) {
        /* Pick the globally best island */
        int best_rank = 0;
        for (int r = 1; r < g_world_size; r++)
            if (all_losses[r] < all_losses[best_rank])
                best_rank = r;

        double global_best_loss = all_losses[best_rank];
        double *global_best_chrom = all_chroms + (size_t)best_rank * n_genes;

        save_sidebyside(target, global_best_chrom, final_file, &prof, &cfg);

        double wall_elapsed = profiler_now() - wall_start;
        logprintf("\nDone. Global best loss=%.6f  (from rank %d)  Saved %s\n",
                  global_best_loss, best_rank, final_file);
        logprintf("Wall time    : %.2f s  (%d generations)\n",
                  wall_elapsed, final_st.generation);

        /* Fix up profiler: optimize bucket should exclude render time */
        prof.totals[BUCKET_OPTIMIZE] -= prof.totals[BUCKET_RENDER];
        profiler_report(&prof, g_log);

        ga_save(&ga, save_file);

        free(all_chroms);
        free(all_losses);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    ga_destroy(&ga);
    cuda_renderer_free();
    image_free(target);
    if (g_log) fclose(g_log);

    MPI_Finalize();
    return 0;
}

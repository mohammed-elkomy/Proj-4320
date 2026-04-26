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

/* ── Save a side-by-side comparison (target | current best) as PPM ─────── */
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
    cuda_renderer_init(target->w, target->h);

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

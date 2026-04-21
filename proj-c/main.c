/*  main.c  —  Simulation driver
 *
 *  Loads target.ppm, runs the GA to approximate it with triangles,
 *  saves PPM snapshots, and prints the timing report.
 *
 *  Output files:
 *      progress_GA_genNNNN.ppm    — snapshot every VISUALISE_EVERY generations
 *      final_result.ppm           — best result at termination
 */

#include "triangle_ga.h"
#include <stdio.h>
#include <string.h>

#define TARGET_FILE     "target.ppm"
#define VISUALISE_EVERY 500

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

    printf("Target image : %d x %d (loaded from %s)\n",
           target->w, target->h, TARGET_FILE);
    printf("State vector : length %d  "
           "(%d triangles x (3 vertices x 2 coords + 4 RGBA))\n",
           N_GENES, N_TRIANGLES);
    printf("Solver       : GA\n\n");

    /* Run GA */
    GA ga;
    ga_init(&ga, target, &prof);

    while (!ga_is_done(&ga)) {
        GAStats st = ga_step(&ga);

        printf("Gen %4d | best=%.6f | avg=%.6f | stagnation=%d/%d\n",
               st.generation, st.best_loss, st.avg_loss,
               ga.stagnation_count, STAGNATION_GENS);

        if (st.generation % VISUALISE_EVERY == 0 || ga_is_done(&ga)) {
            char path[128];
            snprintf(path, sizeof(path),
                     "progress_GA_gen%04d.ppm", st.generation);
            save_sidebyside(target, ga.population[st.best_idx], path, &prof);
        }
    }

    /* Final output */
    GAStats final_st = ga_stats(&ga);
    save_sidebyside(target, ga.population[final_st.best_idx],
                    "final_result.ppm", &prof);

    printf("\nDone. Final loss=%.6f  Saved final_result.ppm\n",
           final_st.best_loss);

    /* Fix up profiler: optimize bucket should exclude render time */
    prof.totals[BUCKET_OPTIMIZE] -= prof.totals[BUCKET_RENDER];

    profiler_report(&prof);

    ga_destroy(&ga);
    image_free(target);

    return 0;
}

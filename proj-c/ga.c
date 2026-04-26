/*  ga.c  —  Genetic Algorithm solver
 *
 *  GA ported from Zsolnai's EvoLisa implementation:
 *    - Two crossover strategies: single-point and uniform-random per vertex
 *    - Two mutation strategies: perturbation (mutate1) and randomisation (mutate2)
 *    - Random parent selection from entire population (post-sort)
 *    - 75% replacement rate, 95% crossover probability
 *    - Alpha channel fixed at 0.15, not mutated
 */

#include "triangle_ga.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Loss ─────────────────────────────────────────────────────────────────── */

double compute_loss(const double *x, const Image *target, Image *scratch,
                    Profiler *prof)
{
    float verts[N_VERTEX_GENES];
    float cols[N_COLOR_GENES];
    vec_to_parts(x, verts, cols);

    render_triangles(verts, cols, scratch, prof);

    int n = target->w * target->h * 3;
    double mse = 0.0;
    const float *a = scratch->data;
    const float *b = target->data;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return mse / n;
}

/* ── Ground truth TESTING ONLY ────────────────────────────────────────────── */

void build_ground_truth(Image *target_out, double *gt_x_out, Profiler *prof)
{
    static const float gt_vertices[] = {
        0.10f, 0.10f,  0.50f, 0.05f,  0.30f, 0.50f,
        0.50f, 0.20f,  0.90f, 0.10f,  0.80f, 0.60f,
        0.10f, 0.50f,  0.40f, 0.90f,  0.05f, 0.95f,
        0.40f, 0.40f,  0.70f, 0.50f,  0.55f, 0.85f,
        0.20f, 0.60f,  0.60f, 0.70f,  0.30f, 0.95f,
    };

    static const float gt_colors[] = {
        0.9f, 0.2f, 0.2f, 0.7f,
        0.2f, 0.7f, 0.3f, 0.7f,
        0.2f, 0.3f, 0.9f, 0.7f,
        0.9f, 0.8f, 0.1f, 0.6f,
        0.7f, 0.1f, 0.8f, 0.6f,
    };

    parts_to_vec(gt_vertices, gt_colors, gt_x_out);
    render_triangles(gt_vertices, gt_colors, target_out, prof);
}

/* ── Sorting helper ───────────────────────────────────────────────────────── */

typedef struct { double fitness; int index; } FitIdx;

static int fit_cmp(const void *a, const void *b) {
    double fa = ((const FitIdx *)a)->fitness;
    double fb = ((const FitIdx *)b)->fitness;
    return (fa > fb) - (fa < fb);
}

/* ── Crossover strategies ─────────────────────────────────────────────────── */

/* Single-point crossover at a triangle boundary — mirrors Zsolnai's crossover1p().
 * cp is a polygon index in [0, N_TRIANGLES]: triangles 0..cp-1 from p1, rest from p2. */
static void crossover1p(const double *p1, const double *p2, double *child) {
    int cp = rng_int(N_TRIANGLES + 1);  /* round(RND2*POLYCOUNT) equivalent */
    for (int i = 0; i < N_TRIANGLES; i++) {
        const double *src = (i < cp) ? p1 : p2;
        for (int k = 0; k < 6; k++)
            child[i*6 + k] = src[i*6 + k];
        int cidx = N_VERTEX_GENES + i*4;
        child[cidx+0] = src[cidx+0];
        child[cidx+1] = src[cidx+1];
        child[cidx+2] = src[cidx+2];
        child[cidx+3] = src[cidx+3];
    }
}

/* Per-vertex uniform crossover — mirrors Zsolnai's crossoverrand().
 * For each triangle vertex, pick a parent at random; color follows the
 * last vertex's parent (matching the original inner-loop assignment). */
static void crossoverrand(const double *p1, const double *p2, double *child) {
    for (int i = 0; i < N_TRIANGLES; i++) {
        for (int j = 0; j < 3; j++) {
            const double *src = (rng_uniform() < 0.5) ? p1 : p2;
            child[i*6 + j*2 + 0] = src[i*6 + j*2 + 0];
            child[i*6 + j*2 + 1] = src[i*6 + j*2 + 1];
            /* Color assignment inside vertex loop — last vertex's parent wins */
            int cidx = N_VERTEX_GENES + i*4;
            child[cidx+0] = src[cidx+0];
            child[cidx+1] = src[cidx+1];
            child[cidx+2] = src[cidx+2];
            child[cidx+3] = src[cidx+3];
        }
    }
}

/* ── Mutation strategies ──────────────────────────────────────────────────── */

/* Small perturbations scaled by 1/perturbation; out-of-range genes reset to
 * uniform random (matching Zsolnai's mutate1).  When |perturbation| is near 0,
 * steps are huge and genes reset — intentional coarse-to-fine effect.
 * perturbation=0 is a no-op to avoid division by zero. */
static void mutate1(double *chrom, double perturbation) {
    if (perturbation == 0.0) return;
    for (int j = 0; j < N_TRIANGLES; j++) {
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 2; l++) {
                int idx = j*6 + k*2 + l;
                if (rng_uniform() < 0.25) {
                    chrom[idx] += (2.0*rng_uniform() - 1.0) / perturbation;
                    if (chrom[idx] < 0.0 || chrom[idx] > 1.0)
                        chrom[idx] = rng_uniform();
                }
            }
        }
        /* Perturb RGB only; alpha is left untouched */
        int cidx = N_VERTEX_GENES + j*4;
        if (rng_uniform() < 0.5) {
            chrom[cidx+0] += 10.0 * (2.0*rng_uniform()-1.0) / perturbation;
            chrom[cidx+1] += 10.0 * (2.0*rng_uniform()-1.0) / perturbation;
            chrom[cidx+2] += 10.0 * (2.0*rng_uniform()-1.0) / perturbation;
        }
        for (int c = 0; c < 3; c++) {
            if (chrom[cidx+c] < 0.0 || chrom[cidx+c] > 1.0)
                chrom[cidx+c] = rng_uniform();
        }
    }
}

/* Randomise ~50% of vertex and RGB genes from scratch (mirrors Zsolnai's mutate2) */
static void mutate2(double *chrom) {
    for (int j = 0; j < N_TRIANGLES; j++) {
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 2; l++) {
                if (rng_uniform() > 0.5)
                    chrom[j*6 + k*2 + l] = rng_uniform();
            }
        }
        int cidx = N_VERTEX_GENES + j*4;
        if (rng_uniform() > 0.5) {
            chrom[cidx+0] = rng_uniform();
            chrom[cidx+1] = rng_uniform();
            chrom[cidx+2] = rng_uniform();
            /* alpha stays fixed */
        }
    }
}

/* ── GA lifetime ──────────────────────────────────────────────────────────── */

void ga_init(GA *ga, Image *target, Profiler *prof)
{
    ga->target  = target;
    ga->prof    = prof;
    ga->scratch = image_alloc(target->w, target->h);

    ga->generation       = 0;
    ga->best_loss        = DBL_MAX;
    ga->stagnation_count = 0;
    ga->done             = 0;

    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < N_TRIANGLES; j++) {
            for (int k = 0; k < 3*2; k++)
                ga->population[i][j*6 + k] = rng_uniform();
            int cidx = N_VERTEX_GENES + j*4;
            ga->population[i][cidx+0] = rng_uniform();
            ga->population[i][cidx+1] = rng_uniform();
            ga->population[i][cidx+2] = rng_uniform();
            ga->population[i][cidx+3] = ALPHA_INIT;
        }
    }

    batch_compute_loss_gpu((const double *)ga->population, POP_SIZE,
                           ga->fitnesses, target->w, target->h);

    for (int i = 0; i < POP_SIZE; i++)
        if (ga->fitnesses[i] < ga->best_loss)
            ga->best_loss = ga->fitnesses[i];
}

void ga_destroy(GA *ga) {
    image_free(ga->scratch);
    ga->scratch = NULL;
}

int ga_is_done(const GA *ga) { return ga->done; }

/* ── One generation ───────────────────────────────────────────────────────── */

GAStats ga_step(GA *ga)
{
    if (ga->done) return ga_stats(ga);

    double t0 = profiler_now();

    /* 1. Sort ascending by fitness */
    FitIdx order[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) {
        order[i].fitness = ga->fitnesses[i];
        order[i].index   = i;
    }
    qsort(order, POP_SIZE, sizeof(FitIdx), fit_cmp);

    double tmp_pop[POP_SIZE][N_GENES];
    double tmp_fit[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) {
        memcpy(tmp_pop[i], ga->population[order[i].index], N_GENES * sizeof(double));
        tmp_fit[i] = order[i].fitness;
    }
    memcpy(ga->population, tmp_pop, sizeof(tmp_pop));
    memcpy(ga->fitnesses,  tmp_fit, sizeof(tmp_fit));

    /* 2. Replace bottom DISC_COUNT chromosomes.
     *    elite_boundary = POP_SIZE - DISC_COUNT; indices > elite_boundary replaced.
     *    Parents selected randomly from the entire sorted population so fitter
     *    individuals are still more likely to be picked on average.             */
    const int elite_boundary = POP_SIZE - DISC_COUNT;
    for (int i = 0; i < POP_SIZE; i++) {
        if (i > elite_boundary) {
            if (rng_uniform() < CROSSOVER_PROB) {
                int ind1 = rng_int(POP_SIZE);
                int ind2 = rng_int(POP_SIZE);
                if (rng_uniform() < 0.5)
                    crossover1p(ga->population[ind1], ga->population[ind2],
                                ga->population[i]);
                else
                    crossoverrand(ga->population[ind1], ga->population[ind2],
                                  ga->population[i]);
            } else {
                /* 95% small perturbation, 5% full randomisation */
                if (rng_uniform() < 0.95)
                    mutate1(ga->population[i],
                            500.0 * (2.0*rng_uniform() - 1.0));
                else
                    mutate2(ga->population[i]);
            }
        }
    }

    /* 3. Re-evaluate fitness for replaced chromosomes only */
    int n_replaced = POP_SIZE - elite_boundary - 1;
    if (n_replaced > 0)
        batch_compute_loss_gpu((const double *)ga->population + (elite_boundary + 1) * N_GENES,
                               n_replaced,
                               ga->fitnesses + elite_boundary + 1,
                               ga->target->w, ga->target->h);

    ga->generation++;

    /* 4. Stagnation check */
    double current_best = ga->fitnesses[0];
    for (int i = 1; i < POP_SIZE; i++)
        if (ga->fitnesses[i] < current_best)
            current_best = ga->fitnesses[i];

    int improved = STAGNATION_RELATIVE
                   ? (current_best < ga->best_loss * (1.0 - STAGNATION_REL_TOL))
                   : (current_best < ga->best_loss - STAGNATION_ABS_TOL);

    if (improved) {
        ga->best_loss        = current_best;
        ga->stagnation_count = 0;
    } else {
        ga->stagnation_count++;
    }

    if (ga->stagnation_count >= STAGNATION_GENS) {
        ga->done = 1;
        printf("[GA] Stagnation after %d generations. Stopping.\n",
               ga->generation);
    }


    profiler_add(ga->prof, BUCKET_OPTIMIZE, profiler_now() - t0);

    return ga_stats(ga);
}

int ga_save(const GA *ga, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    fwrite(&ga->generation,       sizeof(int),    1,        f);
    fwrite(&ga->best_loss,        sizeof(double), 1,        f);
    fwrite(&ga->stagnation_count, sizeof(int),    1,        f);
    fwrite(ga->population,        sizeof(double), POP_SIZE * N_GENES, f);
    fwrite(ga->fitnesses,         sizeof(double), POP_SIZE, f);

    fclose(f);
    return 0;
}

int ga_load(GA *ga, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;   /* file doesn't exist — not an error, just start fresh */

    int ok = 1;
    ok &= (fread(&ga->generation,       sizeof(int),    1,                  f) == 1);
    ok &= (fread(&ga->best_loss,        sizeof(double), 1,                  f) == 1);
    ok &= (fread(&ga->stagnation_count, sizeof(int),    1,                  f) == 1);
    ok &= (fread(ga->population,        sizeof(double), POP_SIZE * N_GENES, f) == POP_SIZE * N_GENES);
    ok &= (fread(ga->fitnesses,         sizeof(double), POP_SIZE,           f) == POP_SIZE);

    fclose(f);

    if (!ok) {
        fprintf(stderr, "ga_load: checkpoint file '%s' is corrupted — starting fresh.\n", path);
        return -1;
    }

    ga->done = 0;
    return 0;
}

GAStats ga_stats(const GA *ga)
{
    GAStats s;
    s.generation = ga->generation;
    s.best_loss  = DBL_MAX;
    s.avg_loss   = 0.0;
    s.best_idx   = 0;

    for (int i = 0; i < POP_SIZE; i++) {
        s.avg_loss += ga->fitnesses[i];
        if (ga->fitnesses[i] < s.best_loss) {
            s.best_loss = ga->fitnesses[i];
            s.best_idx  = i;
        }
    }
    s.avg_loss /= POP_SIZE;
    return s;
}

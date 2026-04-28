/*  ga.c  —  Genetic Algorithm solver */

#include "app.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Loss helpers ─────────────────────────────────────────────────────────── */

static double loss_mse(const float *rendered, const float *target, int n)
{
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)rendered[i] - (double)target[i];
        acc += d * d;
    }
    return acc / n;
}

/* Quartic (L4) loss: sum(d^4)/n.  A pixel twice as wrong costs 16× instead
 * of MSE's 4×, aggressively penalising the worst-case errors. */
static double loss_l4(const float *rendered, const float *target, int n)
{
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)rendered[i] - (double)target[i];
        double d2 = d * d;
        acc += d2 * d2;
    }
    return acc / n;
}

/* Per-channel RGB SSIM loss = 1 − mean(SSIM_R, SSIM_G, SSIM_B).
 * Computes SSIM independently for each colour channel so that hue errors
 * are penalised — luminance-only SSIM lets the GA match brightness while
 * getting colours completely wrong, producing a colour-negative appearance.
 * Uses a single global patch (whole image) for the CPU fallback path. */
static double loss_ssim(const float *rendered, const float *target, int w, int h)
{
    const double C1 = 1e-4, C2 = 9e-4;
    double N = (double)(w * h);
    double ssim_sum = 0.0;

    for (int ch = 0; ch < 3; ch++) {
        double sr=0, st=0, sr2=0, st2=0, srt=0;
        for (int i = 0; i < (int)N; i++) {
            double r = rendered[i*3 + ch];
            double t = target  [i*3 + ch];
            sr += r; st += t; sr2 += r*r; st2 += t*t; srt += r*t;
        }
        double mu_r = sr/N, mu_t = st/N;
        double var_r = sr2/N - mu_r*mu_r;
        double var_t = st2/N - mu_t*mu_t;
        double cov   = srt/N - mu_r*mu_t;
        ssim_sum += ((2*mu_r*mu_t + C1) * (2*cov + C2))
                  / ((mu_r*mu_r + mu_t*mu_t + C1) * (var_r + var_t + C2));
    }
    return 1.0 - ssim_sum / 3.0;
}

/* Binary cross-entropy (log-likelihood) per channel: mean(-t*log(p+e)-(1-t)*log(1-p+e)).         
 * Treats each channel as a Bernoulli probability — more aggressive than MSE near 0/1 extremes. */
static double loss_logll(const float *rendered, const float *target, int n)
{
    const double eps = 1e-7;
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double p = (double)rendered[i];
        double t = (double)target[i];
        acc += -(t * log(p + eps) + (1.0 - t) * log(1.0 - p + eps));
    }
    return acc / n;
}

/* Area-weighted MSE: MSE * (total_normalised_triangle_area + e)^power.
 * Vertices are in [0,1]^2 space; area is computed there so w/h cancel.
 * Penalises large triangles — the GA is pushed toward finer coverage. */
static double loss_wmse(const float *rendered, const float *target, int n,
                        const float *vertices, int n_triangles, double power)
{
    double mse = loss_mse(rendered, target, n);
    double norm_area = 0.0;
    for (int t = 0; t < n_triangles; t++) {
        double ax = vertices[t*6+2] - vertices[t*6+0];
        double ay = vertices[t*6+3] - vertices[t*6+1];
        double bx = vertices[t*6+4] - vertices[t*6+0];
        double by = vertices[t*6+5] - vertices[t*6+1];
        norm_area += fabs(ax*by - ay*bx) * 0.5;
    }
    return mse * (1.0 + pow(norm_area, power));
}

/* ── Loss ─────────────────────────────────────────────────────────────────── */

double compute_loss(const double *x, const Image *target, Image *scratch,
                    Profiler *prof, const AppConfig *cfg)
{
    float *verts = malloc(cfg->n_vertex_genes * sizeof(float));
    float *cols  = malloc(cfg->n_color_genes  * sizeof(float));
    vec_to_parts(x, verts, cols, cfg);

    render_triangles(verts, cols, scratch, prof);

    int n = target->w * target->h * 3;
    double result;
    if (cfg->loss_type == LOSS_SSIM)
        result = loss_ssim(scratch->data, target->data, target->w, target->h);
    else if (cfg->loss_type == LOSS_L4)
        result = loss_l4(scratch->data, target->data, n);
    else if (cfg->loss_type == LOSS_LOGLL)
        result = loss_logll(scratch->data, target->data, n);
    else if (cfg->loss_type == LOSS_WMSE)
        result = loss_wmse(scratch->data, target->data, n,
                           verts, cfg->n_triangles, cfg->wmse_power);
    else
        result = loss_mse(scratch->data, target->data, n);

    free(verts);
    free(cols);
    return result;
}

/* ── Ground truth TESTING ONLY ────────────────────────────────────────────── */

void build_ground_truth(Image *target_out, double *gt_x_out,
                        Profiler *prof, const AppConfig *cfg)
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
    parts_to_vec(gt_vertices, gt_colors, gt_x_out, cfg);
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

static void crossover1p(const double *p1, const double *p2, double *child,
                        int n_triangles, int n_vertex_genes)
{
    int cp = rng_int(n_triangles + 1);
    for (int i = 0; i < n_triangles; i++) {
        const double *src = (i < cp) ? p1 : p2;
        for (int k = 0; k < 6; k++)
            child[i*6 + k] = src[i*6 + k];
        int cidx = n_vertex_genes + i*4;
        child[cidx+0] = src[cidx+0];
        child[cidx+1] = src[cidx+1];
        child[cidx+2] = src[cidx+2];
        child[cidx+3] = src[cidx+3];
    }
}

static void crossoverrand(const double *p1, const double *p2, double *child,
                          int n_triangles, int n_vertex_genes)
{
    for (int i = 0; i < n_triangles; i++) {
        for (int j = 0; j < 3; j++) {
            const double *src = (rng_uniform() < 0.5) ? p1 : p2;
            child[i*6 + j*2 + 0] = src[i*6 + j*2 + 0];
            child[i*6 + j*2 + 1] = src[i*6 + j*2 + 1];
            int cidx = n_vertex_genes + i*4;
            child[cidx+0] = src[cidx+0];
            child[cidx+1] = src[cidx+1];
            child[cidx+2] = src[cidx+2];
            child[cidx+3] = src[cidx+3];
        }
    }
}

/* ── Mutation strategies ──────────────────────────────────────────────────── */

static void mutate1(double *chrom, double perturbation,
                    int n_triangles, int n_vertex_genes)
{
    if (perturbation == 0.0) return;
    for (int j = 0; j < n_triangles; j++) {
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
        int cidx = n_vertex_genes + j*4;
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

static void mutate2(double *chrom, int n_triangles, int n_vertex_genes) {
    for (int j = 0; j < n_triangles; j++) {
        for (int k = 0; k < 3; k++)
            for (int l = 0; l < 2; l++)
                if (rng_uniform() > 0.5)
                    chrom[j*6 + k*2 + l] = rng_uniform();
        int cidx = n_vertex_genes + j*4;
        if (rng_uniform() > 0.5) {
            chrom[cidx+0] = rng_uniform();
            chrom[cidx+1] = rng_uniform();
            chrom[cidx+2] = rng_uniform();
        }
    }
}

/* ── GA lifetime ──────────────────────────────────────────────────────────── */

void ga_init(GA *ga, Image *target, Profiler *prof, const AppConfig *cfg)
{
    ga->target  = target;
    ga->prof    = prof;
    ga->cfg     = cfg;
    ga->scratch = image_alloc(target->w, target->h);

    ga->generation       = 0;
    ga->best_loss        = DBL_MAX;
    ga->stagnation_count = 0;
    ga->done             = 0;

    ga->population = malloc((size_t)cfg->pop_size * cfg->n_genes * sizeof(double));
    ga->fitnesses  = malloc((size_t)cfg->pop_size * sizeof(double));

    for (int i = 0; i < cfg->pop_size; i++) {
        double *chrom = GA_CHROM(ga, i);
        for (int j = 0; j < cfg->n_triangles; j++) {
            for (int k = 0; k < 3*2; k++)
                chrom[j*6 + k] = rng_uniform();
            int cidx = cfg->n_vertex_genes + j*4;
            chrom[cidx+0] = rng_uniform();
            chrom[cidx+1] = rng_uniform();
            chrom[cidx+2] = rng_uniform();
            chrom[cidx+3] = cfg->alpha_init;
        }
    }

    batch_compute_loss_gpu(ga->population, cfg->pop_size,
                           ga->fitnesses, target->w, target->h, prof,
                           cfg->loss_type, cfg->wmse_power);

    for (int i = 0; i < cfg->pop_size; i++)
        if (ga->fitnesses[i] < ga->best_loss)
            ga->best_loss = ga->fitnesses[i];
}

void ga_destroy(GA *ga) {
    image_free(ga->scratch);
    free(ga->population);
    free(ga->fitnesses);
    ga->scratch    = NULL;
    ga->population = NULL;
    ga->fitnesses  = NULL;
}

int ga_is_done(const GA *ga) { return ga->done; }

/* ── One generation ───────────────────────────────────────────────────────── */

GAStats ga_step(GA *ga)
{
    if (ga->done) return ga_stats(ga);

    const AppConfig *cfg = ga->cfg;
    int pop_size      = cfg->pop_size;
    int n_genes       = cfg->n_genes;
    int n_triangles   = cfg->n_triangles;
    int n_vertex_genes = cfg->n_vertex_genes;
    int disc_count    = cfg->disc_count;

    double t0 = profiler_now();

    /* 1. Sort ascending by fitness */
    FitIdx *order   = malloc(pop_size * sizeof(FitIdx));
    double *tmp_pop = malloc((size_t)pop_size * n_genes * sizeof(double));
    double *tmp_fit = malloc(pop_size * sizeof(double));

    for (int i = 0; i < pop_size; i++) {
        order[i].fitness = ga->fitnesses[i];
        order[i].index   = i;
    }
    qsort(order, pop_size, sizeof(FitIdx), fit_cmp);

    for (int i = 0; i < pop_size; i++) {
        memcpy(tmp_pop + (size_t)i * n_genes,
               GA_CHROM(ga, order[i].index),
               n_genes * sizeof(double));
        tmp_fit[i] = order[i].fitness;
    }
    memcpy(ga->population, tmp_pop, (size_t)pop_size * n_genes * sizeof(double));
    memcpy(ga->fitnesses,  tmp_fit, pop_size * sizeof(double));

    free(order);
    free(tmp_pop);
    free(tmp_fit);

    /* 2. Replace bottom disc_count chromosomes */
    const int elite_boundary = pop_size - disc_count;
    for (int i = 0; i < pop_size; i++) {
        if (i > elite_boundary) {
            double *child = GA_CHROM(ga, i);
            if (rng_uniform() < cfg->crossover_prob) {
                const double *p1 = GA_CHROM(ga, rng_int(pop_size));
                const double *p2 = GA_CHROM(ga, rng_int(pop_size));
                if (rng_uniform() < 0.5)
                    crossover1p(p1, p2, child, n_triangles, n_vertex_genes);
                else
                    crossoverrand(p1, p2, child, n_triangles, n_vertex_genes);
            } else {
                if (rng_uniform() < 0.95)
                    mutate1(child, 500.0 * (2.0*rng_uniform() - 1.0),
                            n_triangles, n_vertex_genes);
                else
                    mutate2(child, n_triangles, n_vertex_genes);
            }
        }
    }

    /* 3. Re-evaluate fitness for replaced chromosomes only */
    int n_replaced = pop_size - elite_boundary - 1;
    if (n_replaced > 0)
        batch_compute_loss_gpu(
            ga->population + (size_t)(elite_boundary + 1) * n_genes,
            n_replaced,
            ga->fitnesses + elite_boundary + 1,
            ga->target->w, ga->target->h, ga->prof,
            cfg->loss_type, cfg->wmse_power);

    ga->generation++;

    /* 4. Termination checks */
    double current_best = ga->fitnesses[0];
    for (int i = 1; i < pop_size; i++)
        if (ga->fitnesses[i] < current_best)
            current_best = ga->fitnesses[i];

    int improved = cfg->stagnation_relative
                   ? (current_best < ga->best_loss * (1.0 - cfg->stagnation_rel_tol))
                   : (current_best < ga->best_loss - cfg->stagnation_abs_tol);

    if (improved) {
        ga->best_loss        = current_best;
        ga->stagnation_count = 0;
    } else {
        ga->stagnation_count++;
    }

    if (cfg->max_generations == 0 && cfg->stagnation_gens > 0
            && ga->stagnation_count >= cfg->stagnation_gens) {
        ga->done = 1;
        printf("[GA] Stagnation after %d generations. Stopping.\n", ga->generation);
    }

    if (cfg->max_generations > 0 && ga->generation >= cfg->max_generations) {
        ga->done = 1;
        printf("[GA] Generation limit (%d) reached. Stopping.\n", cfg->max_generations);
    }

    profiler_add(ga->prof, BUCKET_OPTIMIZE, profiler_now() - t0);

    return ga_stats(ga);
}

int ga_save(const GA *ga, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    int pop_size = ga->cfg->pop_size;
    int n_genes  = ga->cfg->n_genes;
    fwrite(&ga->generation,       sizeof(int),    1,                         f);
    fwrite(&ga->best_loss,        sizeof(double), 1,                         f);
    fwrite(&ga->stagnation_count, sizeof(int),    1,                         f);
    fwrite(ga->population,        sizeof(double), (size_t)pop_size * n_genes, f);
    fwrite(ga->fitnesses,         sizeof(double), pop_size,                   f);

    fclose(f);
    return 0;
}

int ga_load(GA *ga, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    int pop_size = ga->cfg->pop_size;
    int n_genes  = ga->cfg->n_genes;
    int ok = 1;
    ok &= (fread(&ga->generation,       sizeof(int),    1,                         f) == 1);
    ok &= (fread(&ga->best_loss,        sizeof(double), 1,                         f) == 1);
    ok &= (fread(&ga->stagnation_count, sizeof(int),    1,                         f) == 1);
    ok &= (fread(ga->population,        sizeof(double), (size_t)pop_size * n_genes, f) == (size_t)pop_size * n_genes);
    ok &= (fread(ga->fitnesses,         sizeof(double), pop_size,                   f) == (size_t)pop_size);

    fclose(f);

    if (!ok) {
        fprintf(stderr, "ga_load: '%s' is corrupted or was made with different N_TRIANGLES/POP_SIZE — starting fresh.\n", path);
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

    for (int i = 0; i < ga->cfg->pop_size; i++) {
        s.avg_loss += ga->fitnesses[i];
        if (ga->fitnesses[i] < s.best_loss) {
            s.best_loss = ga->fitnesses[i];
            s.best_idx  = i;
        }
    }
    s.avg_loss /= ga->cfg->pop_size;
    return s;
}

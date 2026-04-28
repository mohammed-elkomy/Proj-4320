#ifndef APP_H
#define APP_H

#include <stdint.h>
#include <stdio.h>

/* Allow this header to be included from both C and C++ (nvcc) translation units */
#ifdef __cplusplus
extern "C" {
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  Profiler buckets  (compile-time only — not user-facing)
 * ══════════════════════════════════════════════════════════════════════════ */
#define BUCKET_RENDER   0
#define BUCKET_OPTIMIZE 1
#define BUCKET_COUNT    2

/* ══════════════════════════════════════════════════════════════════════════
 *  Timing — dual-arch
 *
 *  On x86_64 (developer laptops): RDTSC serialised with LFENCE.
 *  On POWER9 (the cluster):       fixed 512 MHz timebase via mftbu/mftb.
 *  The correct branch is selected automatically at compile time.
 * ══════════════════════════════════════════════════════════════════════════ */
typedef unsigned long long ticks;

#if defined(__x86_64__)
static __inline__ ticks getticks(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) : : "memory");
    return ((ticks)hi << 32) | lo;
}
static __inline__ double ticks_to_seconds(ticks delta) {
    return (double)delta / 3.0e9;
}
#else
/* POWER9 — fixed 512 MHz timebase register */
static __inline__ ticks getticks(void) {
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb  %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return (((ticks)tbu0) << 32) | tbl;
}
static __inline__ double ticks_to_seconds(ticks delta) {
    return (double)delta / 512000000.0;
}
#endif

static __inline__ ticks  start_timer(void)       { return getticks(); }
static __inline__ double stop_timer(ticks start)  { return ticks_to_seconds(getticks() - start); }

/* ══════════════════════════════════════════════════════════════════════════
 *  Profiler
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double totals[BUCKET_COUNT];
    int    counts[BUCKET_COUNT];
} Profiler;

void   profiler_init(Profiler *p);
double profiler_now(void);
void   profiler_add(Profiler *p, int bucket, double elapsed);
void   profiler_report(Profiler *p, FILE *log); /* log may be NULL */

/* ══════════════════════════════════════════════════════════════════════════
 *  Loss function identifiers
 *
 *  LOSS_MSE  Mean squared error: mean((pred - target)^2) per channel.
 *            Balanced; does not strongly penalise the worst pixel errors.
 *
 *  LOSS_L4   Quartic (L4) loss: mean((pred - target)^4) per channel.
 *            Penalises large errors with the 4th power — far more aggressive
 *            than MSE.  A pixel that is twice as wrong costs 2^4 = 16× more
 *            instead of MSE's 2^2 = 4×.  Drives the worst-case pixels down
 *            at the cost of caring less about near-zero residuals.
 *
 *  LOSS_SSIM Per-channel RGB SSIM loss = 1 − mean(SSIM_R, SSIM_G, SSIM_B).
 *            SSIM jointly measures mean (luminance), variance (contrast), and
 *            cross-covariance (structure) within each 16×16 pixel patch.
 *            Applied independently to R, G, B so that colour errors are
 *            penalised — luminance-only SSIM lets the GA match brightness
 *            while getting hues wrong, producing a colour-negative appearance.
 *            Range [0, 2]; perfect match → 0.
 *
 *  LOSS_LOGLL  Binary cross-entropy (log-likelihood) per RGB channel:
 *              mean(-t·log(p+ε) - (1-t)·log(1-p+ε)).
 *              Treats each rendered pixel channel as a probability in [0,1]
 *              and computes BCE against the target.  More aggressive than MSE
 *              near channel extremes: predicting 0.01 for a target of 1.0
 *              costs log(0.01)≈4.6 nats vs MSE's 0.98²≈0.96.
 *
 *  LOSS_WMSE   Area-weighted MSE: MSE × (total_triangle_area)^WMSE_POWER.
 *              Total area is the sum of triangle areas in normalised [0,1]²
 *              space (can exceed 1 when triangles overlap or are large).
 *              Penalises chromosomes with large triangles, pushing the GA
 *              toward finer-grained coverage while still minimising pixel error.
 *              WMSE_POWER (default 0.5) dampens the area weight so it does not
 *              overwhelm the MSE signal.
 * ══════════════════════════════════════════════════════════════════════════ */
#define LOSS_MSE   0
#define LOSS_L4    1
#define LOSS_SSIM  2
#define LOSS_LOGLL 3
#define LOSS_WMSE  4

/* ══════════════════════════════════════════════════════════════════════════
 *  Runtime configuration
 *
 *  Every field maps directly to a KEY = value line in the config file.
 *  Fields not present in the file keep the defaults set by app_config_init.
 *  After loading, call app_config_finalize to compute derived fields
 *  (n_vertex_genes, n_color_genes, n_genes) and resolve disc_count.
 *
 *  ── Tunable ──────────────────────────────────────────────────────────────
 *  N_TRIANGLES       Triangles used to approximate the image.
 *  POP_SIZE          Chromosomes in the population.
 *  DISC_COUNT        Slots replaced per generation. 0 = auto (ceil(pop*0.75)).
 *  MAX_GENERATIONS   Hard limit. 0 = run until stagnation only. When > 0,
 *                    stagnation check is disabled — exactly this many gens run.
 *  STAGNATION_GENS   Stop after N consecutive gens with no improvement.
 *                    0 = no stagnation limit. Ignored when MAX_GENERATIONS > 0.
 *  STAGNATION_RELATIVE  0 = absolute threshold, 1 = relative threshold.
 *  STAGNATION_ABS_TOL   Min raw improvement to reset stagnation counter.
 *  STAGNATION_REL_TOL   Min fractional improvement (relative mode).
 *  CROSSOVER_PROB    P(crossover) for a discarded slot; else mutation.
 *  ALPHA_INIT        Fixed alpha per triangle (not mutated).
 *  VISUALISE_EVERY   Save side-by-side PPM every N generations.
 *  RUN_PREFIX        Label prepended to the timestamped output folder.
 *  TARGET_FILE       Path to the target PPM image (default: input/target.ppm).
 *  LOSS_TYPE         Loss function: "mse" (0), "l4" (1), "ssim" (2),
 *                    "logll" (3), or "wmse" (4).
 *                    Accepts either the string name or the numeric index.
 *  WMSE_POWER        Exponent for area weight in LOSS_WMSE (default 0.5).
 *                    weight = (total_normalised_area + ε)^WMSE_POWER.
 *                    Larger values amplify area penalty; 0 → pure MSE.
 *  NUM_GPUS          GPUs to use for batch fitness evaluation. 0 = all available.
 *
 *  ── Derived (set by app_config_finalize, not parsed from file) ────────────
 *  n_vertex_genes    N_TRIANGLES * 6
 *  n_color_genes     N_TRIANGLES * 4
 *  n_genes           n_vertex_genes + n_color_genes
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    /* problem size */
    int    n_triangles;
    int    pop_size;
    int    disc_count;         /* 0 = auto */

    /* GA behaviour */
    int    max_generations;
    int    stagnation_gens;
    int    stagnation_relative;
    double stagnation_rel_tol;
    double stagnation_abs_tol;
    double crossover_prob;
    double alpha_init;

    /* loss */
    int    loss_type;          /* LOSS_MSE, LOSS_L4, LOSS_SSIM, LOSS_LOGLL, LOSS_WMSE */
    double wmse_power;         /* exponent for area weight in LOSS_WMSE; default 0.5   */

    /* GPU */
    int    num_gpus;           /* 0 = use all available */

    /* output */
    int    visualise_every;
    char   run_prefix[64];
    char   target_file[256];

    /* derived — populated by app_config_finalize, not parsed */
    int    n_vertex_genes;
    int    n_color_genes;
    int    n_genes;
} AppConfig;

void app_config_init(AppConfig *cfg);                          /* set defaults        */
void app_config_finalize(AppConfig *cfg);                      /* compute derived     */
int  app_config_load(AppConfig *cfg, const char *path);        /* 0=ok, -1=not found  */
void app_config_save(const AppConfig *cfg, const char *path);  /* write key=val file  */

/* ══════════════════════════════════════════════════════════════════════════
 *  Image  (H x W x 3, float32, row-major)
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int    w, h;
    float *data;    /* h * w * 3 floats, RGB [0,1] */
} Image;

Image *image_alloc(int w, int h);
void   image_free(Image *img);
Image *image_read_ppm(const char *path);
int    image_write_ppm(const Image *img, const char *path);

/* ══════════════════════════════════════════════════════════════════════════
 *  Rendering
 * ══════════════════════════════════════════════════════════════════════════ */
void render_triangles(const float *vertices_flat, const float *colors,
                      Image *dst, Profiler *prof);

/* ══════════════════════════════════════════════════════════════════════════
 *  State vector helpers
 * ══════════════════════════════════════════════════════════════════════════ */
void vec_to_parts(const double *x, float *vertices_flat, float *colors,
                  const AppConfig *cfg);
void parts_to_vec(const float *vertices_flat, const float *colors, double *x,
                  const AppConfig *cfg);

/* ══════════════════════════════════════════════════════════════════════════
 *  Loss
 * ══════════════════════════════════════════════════════════════════════════ */
double compute_loss(const double *x, const Image *target, Image *scratch,
                    Profiler *prof, const AppConfig *cfg);

/* ══════════════════════════════════════════════════════════════════════════
 *  Ground truth (testing only — not used in normal runs)
 * ══════════════════════════════════════════════════════════════════════════ */
void build_ground_truth(Image *target_out, double *gt_x_out,
                        Profiler *prof, const AppConfig *cfg);

/* ══════════════════════════════════════════════════════════════════════════
 *  Genetic Algorithm
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int    generation;
    double best_loss;
    double avg_loss;
    int    best_idx;
} GAStats;

typedef struct {
    double  *population;       /* [pop_size * n_genes], row-major, heap-allocated */
    double  *fitnesses;        /* [pop_size], heap-allocated                      */
    Image   *target;
    Image   *scratch;
    Profiler        *prof;
    const AppConfig *cfg;

    int     generation;
    double  best_loss;
    int     stagnation_count;
    int     done;
} GA;

/* Access chromosome i as a pointer to its n_genes doubles */
#define GA_CHROM(ga, i) ((ga)->population + (size_t)(i) * (ga)->cfg->n_genes)

void    ga_init(GA *ga, Image *target, Profiler *prof, const AppConfig *cfg);
void    ga_destroy(GA *ga);
int     ga_is_done(const GA *ga);
GAStats ga_step(GA *ga);
GAStats ga_stats(const GA *ga);
int     ga_save(const GA *ga, const char *path);
int     ga_load(GA *ga, const char *path);

/* ══════════════════════════════════════════════════════════════════════════
 *  RNG  (xoshiro256**)
 * ══════════════════════════════════════════════════════════════════════════ */
void   rng_seed(uint64_t s);
double rng_uniform(void);
double rng_normal(void);
int    rng_int(int n);

/* ══════════════════════════════════════════════════════════════════════════
 *  CUDA renderer lifecycle
 * ══════════════════════════════════════════════════════════════════════════ */
int  cuda_num_gpus(void);
void cuda_renderer_init(const Image *target, const AppConfig *cfg);
void cuda_renderer_free(void);
void batch_compute_loss_gpu(const double *pop, int count,
                            double *losses_out, int w, int h,
                            Profiler *prof, int loss_type,
                            double wmse_power);

#ifdef __cplusplus
}
#endif

#endif /* APP_H */

#ifndef APP_H
#define APP_H

#include <stdint.h>

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
void   profiler_report(Profiler *p);

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

    /* output */
    int    visualise_every;
    char   run_prefix[64];

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
void cuda_renderer_init(const Image *target, const AppConfig *cfg);
void cuda_renderer_free(void);
void batch_compute_loss_gpu(const double *pop, int count,
                            double *losses_out, int w, int h,
                            Profiler *prof);

#ifdef __cplusplus
}
#endif

#endif /* APP_H */

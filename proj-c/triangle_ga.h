#ifndef TRIANGLE_GA_H
#define TRIANGLE_GA_H

#include <stdint.h>

/* ══════════════════════════════════════════════════════════════════════════
 *  Tunable parameters — edit these to control the run
 * ══════════════════════════════════════════════════════════════════════════
 *
 *  N_TRIANGLES     How many triangles are used to approximate the image.
 *                  More = higher quality, slower per generation.
 *                  Zsolnai uses 150; we use 200.
 *
 *  POP_SIZE        Number of chromosomes in the population.
 *                  Larger = more diversity, more fitness evaluations per gen.
 *                  Zsolnai uses 30.
 *
 *  DISC_COUNT      Chromosomes replaced each generation = ceil(POP_SIZE*0.75).
 *                  The top (POP_SIZE - DISC_COUNT) are kept unchanged (elitism).
 *                  Zsolnai: disc = ceil(30*0.75) = 23, keeps 8 elite.
 *
 *  CROSSOVER_PROB  Probability that a new slot gets crossover vs mutation.
 *                  When mutation fires: 95% mutate1 (perturbation), 5% mutate2 (reset).
 *                  Zsolnai uses crp = 0.95.
 *
 *  ALPHA_INIT      Fixed alpha (transparency) for every triangle.
 *                  Not mutated — only inherited via crossover.
 *                  Zsolnai uses 0.15.
 *
 *  STAGNATION_GENS Stop after this many consecutive generations with no
 *                  improvement better than 1e-7. Zsolnai runs forever;
 *                  set to 0 to disable (run until you kill the process).
 *
 *  VISUALISE_EVERY Save a side-by-side PPM snapshot every N generations
 *                  (set in main.c).
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── Problem size ─────────────────────────────────────────────────────────── */
#define N_TRIANGLES  200

#define N_VERTEX_GENES (N_TRIANGLES * 3 * 2)          /* 1200 */
#define N_COLOR_GENES  (N_TRIANGLES * 4)               /*  800 */
#define N_GENES        (N_VERTEX_GENES + N_COLOR_GENES) /* 2000 */

/* ── GA hyper-parameters (Zsolnai defaults) ──────────────────────────────── */
#define POP_SIZE        30
#define DISC_COUNT      23      /* ceil(POP_SIZE * 0.75) */
#define CROSSOVER_PROB  0.95
#define ALPHA_INIT      0.15
#define STAGNATION_GENS 2000

/* ── Profiler buckets ─────────────────────────────────────────────────────── */
#define BUCKET_RENDER   0
#define BUCKET_OPTIMIZE 1
#define BUCKET_COUNT    2

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
 *  Image  (H x W x 3, float32, row-major)
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int    w, h;
    float *data;           /* h * w * 3 floats, RGB [0,1] */
} Image;

Image *image_alloc(int w, int h);
void   image_free(Image *img);
Image *image_read_ppm(const char *path);   /* returns NULL on error */
int    image_write_ppm(const Image *img, const char *path);

/* ══════════════════════════════════════════════════════════════════════════
 *  Rendering
 * ══════════════════════════════════════════════════════════════════════════ */
void render_triangles(const float *vertices_flat, const float *colors,
                      Image *dst, Profiler *prof);

/* ══════════════════════════════════════════════════════════════════════════
 *  State vector helpers
 * ══════════════════════════════════════════════════════════════════════════ */
void vec_to_parts(const double *x, float *vertices_flat, float *colors);
void parts_to_vec(const float *vertices_flat, const float *colors, double *x);

/* ══════════════════════════════════════════════════════════════════════════
 *  Loss
 * ══════════════════════════════════════════════════════════════════════════ */
double compute_loss(const double *x, const Image *target, Image *scratch,
                    Profiler *prof);

/* ══════════════════════════════════════════════════════════════════════════
 *  Ground truth (kept for testing; not used in normal PPM-input runs)
 * ══════════════════════════════════════════════════════════════════════════ */
void build_ground_truth(Image *target_out, double *gt_x_out, Profiler *prof);

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
    double  population[POP_SIZE][N_GENES];
    double  fitnesses[POP_SIZE];
    Image  *target;
    Image  *scratch;
    Profiler *prof;

    int     generation;
    double  best_loss;
    int     stagnation_count;
    int     done;
} GA;

void    ga_init(GA *ga, Image *target, Profiler *prof);
void    ga_destroy(GA *ga);
int     ga_is_done(const GA *ga);
GAStats ga_step(GA *ga);
GAStats ga_stats(const GA *ga);

/* ══════════════════════════════════════════════════════════════════════════
 *  RNG  (xoshiro256**)
 * ══════════════════════════════════════════════════════════════════════════ */
void   rng_seed(uint64_t s);
double rng_uniform(void);   /* [0, 1) */
double rng_normal(void);    /* standard normal */
int    rng_int(int n);      /* [0, n) */

#endif /* TRIANGLE_GA_H */

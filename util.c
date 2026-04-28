/*  util.c  —  RNG, profiler, image alloc / PPM read+write, state vector helpers  */

#include "app.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  RNG  — xoshiro256**
 * ══════════════════════════════════════════════════════════════════════════ */
static uint64_t s_rng[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    const uint64_t result = rotl(s_rng[1] * 5, 7) * 9;
    const uint64_t t = s_rng[1] << 17;
    s_rng[2] ^= s_rng[0];
    s_rng[3] ^= s_rng[1];
    s_rng[1] ^= s_rng[2];
    s_rng[0] ^= s_rng[3];
    s_rng[2] ^= t;
    s_rng[3] = rotl(s_rng[3], 45);
    return result;
}

/* SplitMix64 to seed xoshiro from a single uint64 */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void rng_seed(uint64_t seed) {
    uint64_t sm = seed;
    s_rng[0] = splitmix64(&sm);
    s_rng[1] = splitmix64(&sm);
    s_rng[2] = splitmix64(&sm);
    s_rng[3] = splitmix64(&sm);
}

double rng_uniform(void) {
    return (rng_next() >> 11) * (1.0 / 9007199254740992.0);  /* 2^53 */
}

double rng_normal(void) {
    /* Box-Muller */
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-30) u1 = 1e-30;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int rng_int(int n) {
    return (int)(rng_uniform() * n);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Profiler
 * ══════════════════════════════════════════════════════════════════════════ */
void profiler_init(Profiler *p) {
    memset(p, 0, sizeof(*p));
}

double profiler_now(void) {
    return ticks_to_seconds(getticks());
}

void profiler_add(Profiler *p, int bucket, double elapsed) {
    p->totals[bucket] += elapsed;
    p->counts[bucket] += 1;
}

void profiler_report(Profiler *p, FILE *log) {
    static const char *names[] = { "render", "optimize (net, excl. render)" };

    double total_wall = 0.0;
    for (int i = 0; i < BUCKET_COUNT; i++)
        total_wall += p->totals[i];

#define PROF_PRINT(fmt, ...) \
    do { printf(fmt, ##__VA_ARGS__); if (log) fprintf(log, fmt, ##__VA_ARGS__); } while(0)

    PROF_PRINT("\n");
    PROF_PRINT("--------------------------------------------------------------\n");
    PROF_PRINT("  Timing Report\n");
    PROF_PRINT("--------------------------------------------------------------\n");
    PROF_PRINT("%-32s %10s %8s %10s %10s\n",
               "Bucket", "Total (s)", "Calls", "Mean (ms)", "% of wall");
    PROF_PRINT("--------------------------------------------------------------\n");

    for (int i = 0; i < BUCKET_COUNT; i++) {
        double total   = p->totals[i];
        int    calls   = p->counts[i];
        double mean_ms = calls > 0 ? (total / calls * 1000.0) : 0.0;
        double pct     = total_wall > 0 ? (total / total_wall * 100.0) : 0.0;
        PROF_PRINT("%-32s %10.4f %8d %10.4f %9.1f%%\n",
                   names[i], total, calls, mean_ms, pct);
    }

    PROF_PRINT("--------------------------------------------------------------\n");
    PROF_PRINT("%-32s %10.4f\n", "TOTAL", total_wall);
    PROF_PRINT("--------------------------------------------------------------\n\n");
#undef PROF_PRINT
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Image
 * ══════════════════════════════════════════════════════════════════════════ */
Image *image_alloc(int w, int h) {
    Image *img = (Image *)malloc(sizeof(Image));
    img->w    = w;
    img->h    = h;
    img->data = (float *)calloc((size_t)w * h * 3, sizeof(float));
    return img;
}

void image_free(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

/* ── PPM reader helpers ───────────────────────────────────────────────────── */

static void ppm_skip_ws(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
        } else if (!isspace(c)) {
            ungetc(c, f);
            break;
        }
    }
}

static int ppm_read_int(FILE *f) {
    ppm_skip_ws(f);
    int v = 0, c;
    while ((c = fgetc(f)) != EOF && isdigit(c))
        v = v * 10 + (c - '0');
    if (c != EOF) ungetc(c, f);
    return v;
}

Image *image_read_ppm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    char m0 = (char)fgetc(f), m1 = (char)fgetc(f);
    int is_p6 = (m0 == 'P' && m1 == '6');
    int is_p3 = (m0 == 'P' && m1 == '3');
    if (!is_p6 && !is_p3) {
        fprintf(stderr, "image_read_ppm: not a P3/P6 PPM: %s\n", path);
        fclose(f); return NULL;
    }

    int w      = ppm_read_int(f);
    int h      = ppm_read_int(f);
    int maxval = ppm_read_int(f);
    fgetc(f);  /* consume single whitespace separator before binary/ascii data */

    if (w <= 0 || h <= 0 || maxval <= 0) {
        fprintf(stderr, "image_read_ppm: bad header in %s\n", path);
        fclose(f); return NULL;
    }

    Image *img = image_alloc(w, h);
    float scale = 1.0f / (float)maxval;
    int npix = w * h * 3;

    if (is_p6) {
        for (int i = 0; i < npix; i++) {
            int c = fgetc(f);
            if (c == EOF) {
                fprintf(stderr, "image_read_ppm: premature EOF in %s\n", path);
                image_free(img); fclose(f); return NULL;
            }
            img->data[i] = (float)(unsigned char)c * scale;
        }
    } else {
        for (int i = 0; i < npix; i++)
            img->data[i] = (float)ppm_read_int(f) * scale;
    }

    fclose(f);
    return img;
}

int image_write_ppm(const Image *img, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
    for (int i = 0; i < img->w * img->h * 3; i++) {
        float v = img->data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        uint8_t byte = (uint8_t)(v * 255.0f + 0.5f);
        fputc(byte, f);
    }
    fclose(f);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  State vector helpers
 * ══════════════════════════════════════════════════════════════════════════ */
void vec_to_parts(const double *x, float *vertices_flat, float *colors,
                  const AppConfig *cfg) {
    for (int i = 0; i < cfg->n_vertex_genes; i++)
        vertices_flat[i] = (float)x[i];
    for (int i = 0; i < cfg->n_color_genes; i++)
        colors[i] = (float)x[cfg->n_vertex_genes + i];
}

void parts_to_vec(const float *vertices_flat, const float *colors, double *x,
                  const AppConfig *cfg) {
    for (int i = 0; i < cfg->n_vertex_genes; i++)
        x[i] = (double)vertices_flat[i];
    for (int i = 0; i < cfg->n_color_genes; i++)
        x[cfg->n_vertex_genes + i] = (double)colors[i];
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AppConfig — defaults, finalize, file load, file save
 * ══════════════════════════════════════════════════════════════════════════ */

void app_config_init(AppConfig *cfg) {
    cfg->n_triangles         = 200;
    cfg->pop_size            = 30;
    cfg->disc_count          = 0;     /* 0 = auto: ceil(pop_size * 0.75) */
    cfg->max_generations     = 0;
    cfg->stagnation_gens     = 2000;
    cfg->stagnation_relative = 0;
    cfg->stagnation_rel_tol  = 1e-4;
    cfg->stagnation_abs_tol  = 1e-7;
    cfg->crossover_prob      = 0.95;
    cfg->alpha_init          = 0.15;
    cfg->loss_type           = LOSS_MSE;
    cfg->wmse_power          = 0.5;
    cfg->num_gpus            = 0;      /* 0 = use all available */
    cfg->visualise_every     = 10000;
    strncpy(cfg->run_prefix,  "run",               sizeof(cfg->run_prefix)  - 1);
    cfg->run_prefix[sizeof(cfg->run_prefix) - 1] = '\0';
    strncpy(cfg->target_file, "input/target.ppm",  sizeof(cfg->target_file) - 1);
    cfg->target_file[sizeof(cfg->target_file) - 1] = '\0';
    /* derived fields initialised by finalize */
    cfg->n_vertex_genes = 0;
    cfg->n_color_genes  = 0;
    cfg->n_genes        = 0;
}

void app_config_finalize(AppConfig *cfg) {
    if (cfg->disc_count <= 0)
        cfg->disc_count = (int)(cfg->pop_size * 0.75 + 0.9999); /* ceil */
    cfg->n_vertex_genes = cfg->n_triangles * 3 * 2;
    cfg->n_color_genes  = cfg->n_triangles * 4;
    cfg->n_genes        = cfg->n_vertex_genes + cfg->n_color_genes;
}

static void cfg_trim(char *s) {
    char *p = s;
    while (isspace((unsigned char)*p)) p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    p = s + strlen(s);
    while (p > s && isspace((unsigned char)*(p - 1))) p--;
    *p = '\0';
}

int app_config_load(AppConfig *cfg, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '#' || *p == '\0') continue;

        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = p, *val = eq + 1;
        cfg_trim(key);
        cfg_trim(val);

        if      (!strcmp(key, "N_TRIANGLES"))        cfg->n_triangles        = atoi(val);
        else if (!strcmp(key, "POP_SIZE"))           cfg->pop_size           = atoi(val);
        else if (!strcmp(key, "DISC_COUNT"))         cfg->disc_count         = atoi(val);
        else if (!strcmp(key, "MAX_GENERATIONS"))    cfg->max_generations    = atoi(val);
        else if (!strcmp(key, "STAGNATION_GENS"))    cfg->stagnation_gens    = atoi(val);
        else if (!strcmp(key, "STAGNATION_RELATIVE"))cfg->stagnation_relative= atoi(val);
        else if (!strcmp(key, "STAGNATION_REL_TOL")) cfg->stagnation_rel_tol = atof(val);
        else if (!strcmp(key, "STAGNATION_ABS_TOL")) cfg->stagnation_abs_tol = atof(val);
        else if (!strcmp(key, "CROSSOVER_PROB"))     cfg->crossover_prob     = atof(val);
        else if (!strcmp(key, "ALPHA_INIT"))         cfg->alpha_init         = atof(val);
        else if (!strcmp(key, "VISUALISE_EVERY"))    cfg->visualise_every    = atoi(val);
        else if (!strcmp(key, "NUM_GPUS"))           cfg->num_gpus           = atoi(val);
        else if (!strcmp(key, "LOSS_TYPE")) {
            /* Accept numeric index (0-4) or string name */
            if (val[0] >= '0' && val[0] <= '9') {
                cfg->loss_type = atoi(val);
            } else if (!strcmp(val, "l4")    || !strcmp(val, "L4"))    {
                cfg->loss_type = LOSS_L4;
            } else if (!strcmp(val, "ssim")  || !strcmp(val, "SSIM"))  {
                cfg->loss_type = LOSS_SSIM;
            } else if (!strcmp(val, "logll") || !strcmp(val, "LOGLL")) {
                cfg->loss_type = LOSS_LOGLL;
            } else if (!strcmp(val, "wmse")  || !strcmp(val, "WMSE"))  {
                cfg->loss_type = LOSS_WMSE;
            } else {
                cfg->loss_type = LOSS_MSE;
            }
        }
        else if (!strcmp(key, "WMSE_POWER"))         cfg->wmse_power         = atof(val);
        else if (!strcmp(key, "RUN_PREFIX")) {
            strncpy(cfg->run_prefix, val, sizeof(cfg->run_prefix) - 1);
            cfg->run_prefix[sizeof(cfg->run_prefix) - 1] = '\0';
        }
        else if (!strcmp(key, "TARGET_FILE")) {
            strncpy(cfg->target_file, val, sizeof(cfg->target_file) - 1);
            cfg->target_file[sizeof(cfg->target_file) - 1] = '\0';
        }
    }
    fclose(f);
    return 0;
}

void app_config_save(const AppConfig *cfg, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }

    time_t now = time(NULL);
    fprintf(f, "# Hyperparameters — saved at: %s", ctime(&now));
    fprintf(f, "# Derived (not parsed): n_vertex_genes=%d  n_color_genes=%d  n_genes=%d\n\n",
            cfg->n_vertex_genes, cfg->n_color_genes, cfg->n_genes);
    fprintf(f, "RUN_PREFIX          = %s\n", cfg->run_prefix);
    fprintf(f, "TARGET_FILE         = %s\n\n", cfg->target_file);
    fprintf(f, "N_TRIANGLES         = %d\n", cfg->n_triangles);
    fprintf(f, "POP_SIZE            = %d\n", cfg->pop_size);
    fprintf(f, "DISC_COUNT          = %d\n\n", cfg->disc_count);
    fprintf(f, "MAX_GENERATIONS     = %d\n", cfg->max_generations);
    fprintf(f, "STAGNATION_GENS     = %d\n", cfg->stagnation_gens);
    fprintf(f, "STAGNATION_RELATIVE = %d\n", cfg->stagnation_relative);
    fprintf(f, "STAGNATION_REL_TOL  = %g\n", cfg->stagnation_rel_tol);
    fprintf(f, "STAGNATION_ABS_TOL  = %g\n\n", cfg->stagnation_abs_tol);
    fprintf(f, "CROSSOVER_PROB      = %g\n", cfg->crossover_prob);
    fprintf(f, "ALPHA_INIT          = %g\n", cfg->alpha_init);
    fprintf(f, "LOSS_TYPE           = %s\n",
            cfg->loss_type == LOSS_SSIM  ? "ssim"  :
            cfg->loss_type == LOSS_L4    ? "l4"    :
            cfg->loss_type == LOSS_LOGLL ? "logll" :
            cfg->loss_type == LOSS_WMSE  ? "wmse"  : "mse");
    fprintf(f, "WMSE_POWER          = %g\n", cfg->wmse_power);
    fprintf(f, "NUM_GPUS            = %d\n", cfg->num_gpus);
    fprintf(f, "VISUALISE_EVERY     = %d\n", cfg->visualise_every);
    fclose(f);
}

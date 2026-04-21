/*  util.c  —  RNG, profiler, image alloc / PPM read+write, state vector helpers  */

#define _POSIX_C_SOURCE 199309L

#include "triangle_ga.h"
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
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void profiler_add(Profiler *p, int bucket, double elapsed) {
    p->totals[bucket] += elapsed;
    p->counts[bucket] += 1;
}

void profiler_report(Profiler *p) {
    static const char *names[] = { "render", "optimize (net, excl. render)" };

    double total_wall = 0.0;
    for (int i = 0; i < BUCKET_COUNT; i++)
        total_wall += p->totals[i];

    printf("\n");
    printf("--------------------------------------------------------------\n");
    printf("  Timing Report\n");
    printf("--------------------------------------------------------------\n");
    printf("%-32s %10s %8s %10s %10s\n",
           "Bucket", "Total (s)", "Calls", "Mean (ms)", "% of wall");
    printf("--------------------------------------------------------------\n");

    for (int i = 0; i < BUCKET_COUNT; i++) {
        double total   = p->totals[i];
        int    calls   = p->counts[i];
        double mean_ms = calls > 0 ? (total / calls * 1000.0) : 0.0;
        double pct     = total_wall > 0 ? (total / total_wall * 100.0) : 0.0;
        printf("%-32s %10.4f %8d %10.4f %9.1f%%\n",
               names[i], total, calls, mean_ms, pct);
    }

    printf("--------------------------------------------------------------\n");
    printf("%-32s %10.4f\n", "TOTAL", total_wall);
    printf("--------------------------------------------------------------\n\n");
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
void vec_to_parts(const double *x, float *vertices_flat, float *colors) {
    for (int i = 0; i < N_VERTEX_GENES; i++)
        vertices_flat[i] = (float)x[i];
    for (int i = 0; i < N_COLOR_GENES; i++)
        colors[i] = (float)x[N_VERTEX_GENES + i];
}

void parts_to_vec(const float *vertices_flat, const float *colors, double *x) {
    for (int i = 0; i < N_VERTEX_GENES; i++)
        x[i] = (double)vertices_flat[i];
    for (int i = 0; i < N_COLOR_GENES; i++)
        x[N_VERTEX_GENES + i] = (double)colors[i];
}

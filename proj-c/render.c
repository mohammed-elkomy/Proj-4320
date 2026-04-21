/*  render.c  —  Software triangle rasteriser (no graphics library)
 *
 *  Matches the Python render_triangles() exactly:
 *    - Barycentric point-in-triangle for every pixel
 *    - Ordered alpha blending onto a white canvas
 */

#include "triangle_ga.h"
#include <math.h>
#include <string.h>

void render_triangles(const float *vertices_flat, const float *colors,
                      Image *dst, Profiler *prof)
{
    double t0 = profiler_now();

    const int w = dst->w;
    const int h = dst->h;
    float *canvas = dst->data;

    /* Start with white canvas */
    for (int i = 0; i < w * h * 3; i++)
        canvas[i] = 1.0f;

    for (int t = 0; t < N_TRIANGLES; t++) {
        /* 3 vertices, each (x, y) normalised [0,1] -> pixel coords */
        const float *vbase = vertices_flat + t * 6;
        float x0 = vbase[0] * w, y0 = vbase[1] * h;
        float x1 = vbase[2] * w, y1 = vbase[3] * h;
        float x2 = vbase[4] * w, y2 = vbase[5] * h;

        float r = colors[t * 4 + 0];
        float g = colors[t * 4 + 1];
        float b = colors[t * 4 + 2];
        float a = colors[t * 4 + 3];

        /* Edge vectors for barycentric test (same convention as Python):
         *   v0 = tri[2] - tri[0]
         *   v1 = tri[1] - tri[0]                                          */
        float v0x = x2 - x0, v0y = y2 - y0;
        float v1x = x1 - x0, v1y = y1 - y0;

        float d00 = v0x * v0x + v0y * v0y;
        float d01 = v0x * v1x + v0y * v1y;
        float d11 = v1x * v1x + v1y * v1y;

        float denom = d00 * d11 - d01 * d01;
        if (fabsf(denom) < 1e-10f) continue;
        float inv = 1.0f / denom;

        /* Bounding box (clipped to canvas) to skip obviously-outside pixels */
        float min_x = x0, max_x = x0;
        float min_y = y0, max_y = y0;
        if (x1 < min_x) min_x = x1; if (x1 > max_x) max_x = x1;
        if (x2 < min_x) min_x = x2; if (x2 > max_x) max_x = x2;
        if (y1 < min_y) min_y = y1; if (y1 > max_y) max_y = y1;
        if (y2 < min_y) min_y = y2; if (y2 > max_y) max_y = y2;

        int ix0 = (int)min_x;      if (ix0 < 0) ix0 = 0;
        int iy0 = (int)min_y;      if (iy0 < 0) iy0 = 0;
        int ix1 = (int)(max_x + 1); if (ix1 > w) ix1 = w;
        int iy1 = (int)(max_y + 1); if (iy1 > h) iy1 = h;

        float one_minus_a = 1.0f - a;

        for (int py = iy0; py < iy1; py++) {
            float v2y = (float)py - y0;
            /* Pre-compute partial dot products that only depend on py */
            float d02_y_part = v0x * 0.0f + v0y * v2y;  /* will add v0x * v2x */
            float d12_y_part = v1x * 0.0f + v1y * v2y;
            /* Actually: d02_y_part = v0y * v2y, d12_y_part = v1y * v2y */
            d02_y_part = v0y * v2y;
            d12_y_part = v1y * v2y;

            for (int px = ix0; px < ix1; px++) {
                float v2x = (float)px - x0;

                float d02 = v0x * v2x + d02_y_part;
                float d12 = v1x * v2x + d12_y_part;

                float u = (d11 * d02 - d01 * d12) * inv;
                float v = (d00 * d12 - d01 * d02) * inv;

                if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
                    int idx = (py * w + px) * 3;
                    canvas[idx + 0] = r * a + canvas[idx + 0] * one_minus_a;
                    canvas[idx + 1] = g * a + canvas[idx + 1] * one_minus_a;
                    canvas[idx + 2] = b * a + canvas[idx + 2] * one_minus_a;
                }
            }
        }
    }

    profiler_add(prof, BUCKET_RENDER, profiler_now() - t0);
}

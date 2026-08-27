#include "cubiomes.h"

#include "../cubiomes/finders.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct FloodEntry {
    int i;
    int j;
    int d;
} FloodEntry;

struct Cubiomes {
    Generator g;
    uint8_t *visited;
    size_t visited_capacity;
    uint32_t *candidates;
    size_t candidates_capacity;
    FloodEntry *queue;
    size_t queue_capacity;
};

// getSpline is implemented in the bundled cubiomes biomenoise module, but is
// not part of its public header.
float getSpline(const Spline *sp, const float *vals);

static void ensure_capacity(void **buffer, size_t *capacity, size_t needed,
                            size_t element_size, const char *name) {
    if (*capacity >= needed)
        return;

    size_t new_capacity = *capacity ? *capacity : 4096;
    while (new_capacity < needed) {
        if (new_capacity > SIZE_MAX / 2) {
            new_capacity = needed;
            break;
        }
        new_capacity *= 2;
    }
    if (new_capacity > SIZE_MAX / element_size) {
        fprintf(stderr, "%s is too large\n", name);
        abort();
    }

    void *resized = realloc(*buffer, new_capacity * element_size);
    if (resized == NULL) {
        fprintf(stderr, "could not allocate %s\n", name);
        abort();
    }
    *buffer = resized;
    *capacity = new_capacity;
}

Cubiomes *cubiomes_create(int large_biomes) {
    Cubiomes *cubiomes = calloc(1, sizeof(Cubiomes));
    if (cubiomes == NULL) {
        fprintf(stderr, "cubiomes_create failed\n");
        abort();
    }
    setupGenerator(&cubiomes->g, MC_NEWEST, large_biomes ? LARGE_BIOMES : 0);
    return cubiomes;
}

void cubiomes_free(Cubiomes *cubiomes) {
    free(cubiomes->visited);
    free(cubiomes->candidates);
    free(cubiomes->queue);
    free(cubiomes);
}

void cubiomes_apply_seed(Cubiomes *cubiomes, uint64_t seed) {
    applySeed(&cubiomes->g, DIM_OVERWORLD, seed);
}

static int has_mushroom_continentalness(Generator *g, int scale, int x, int z) {
    int qx = x;
    int qz = z;
    double px;
    double pz;

    if (scale > 4) {
        int qscale = scale / 4;
        int mid = qscale / 2;
        qx = x * qscale + mid;
        qz = z * qscale + mid;
        px = qx;
        pz = qz;
    } else {
        px = qx + sampleDoublePerlin(&g->bn.climate[NP_SHIFT], qx, 0, qz) * 4.0;
        pz = qz + sampleDoublePerlin(&g->bn.climate[NP_SHIFT], qz, qx, 0) * 4.0;
    }

    float c = sampleDoublePerlin(&g->bn.climate[NP_CONTINENTALNESS], px, 0, pz);
    return (int64_t)(10000.0F * c) <= -10500;
}

static int eval(Generator *g, int scale, int x, int y, int z, void *data) {
    (void)y;
    (void)data;
    return has_mushroom_continentalness(g, scale, x, z);
}

static int is_mushroom_fields(Generator *g, int scale, int x, int y, int z) {
    uint64_t dat = 0;
    uint64_t *p_dat = NULL;
    double px;
    double pz;

    if (scale > 4) {
        int qscale = scale / 4;
        int mid = qscale / 2;
        x = x * qscale + mid;
        z = z * qscale + mid;
        px = x;
        pz = z;
        p_dat = &dat;
    } else {
        px = x + sampleDoublePerlin(&g->bn.climate[NP_SHIFT], x, 0, z) * 4.0;
        pz = z + sampleDoublePerlin(&g->bn.climate[NP_SHIFT], z, x, 0) * 4.0;
    }

    float c = sampleDoublePerlin(&g->bn.climate[NP_CONTINENTALNESS], px, 0, pz);
    int64_t c_quantized = (int64_t)(10000.0F * c);
    // Mushroom fields have no parameter point above this continentalness.
    // Rejecting here avoids sampling the other climate fields for most cells.
    if (c_quantized > -10500)
        return 0;

    // This mirrors the remainder of sampleBiomeNoise(), without resampling
    // shifts and continentalness or allocating getBiomeAt()'s one-cell cache.
    float e = sampleDoublePerlin(&g->bn.climate[NP_EROSION], px, 0, pz);
    float w = sampleDoublePerlin(&g->bn.climate[NP_WEIRDNESS], px, 0, pz);
    float np_param[] = {
        c, e, -3.0F * (fabsf(fabsf(w) - 0.6666667F) - 0.33333334F), w,
    };
    double off = getSpline(g->bn.sp, np_param) + 0.015F;
    float d = 1.0 - (y * 4) / 128.0 - 83.0 / 160.0 + off;
    float t = sampleDoublePerlin(&g->bn.climate[NP_TEMPERATURE], px, 0, pz);
    float h = sampleDoublePerlin(&g->bn.climate[NP_HUMIDITY], px, 0, pz);

    int64_t np[] = {
        (int64_t)(10000.0F * t),
        (int64_t)(10000.0F * h),
        c_quantized,
        (int64_t)(10000.0F * e),
        (int64_t)(10000.0F * d),
        (int64_t)(10000.0F * w),
    };
    return climateToBiome(g->mc, (const uint64_t *)np, p_dat) == mushroom_fields;
}

static Range make_range(int32_t x, int32_t z, int32_t range, int32_t scale) {
    return (Range){
        .scale = scale,
        .x = (x - range / 2) / scale,
        .z = (z - range / 2) / scale,
        .sx = range / scale,
        .sz = range / scale,
        .y = 256 / scale,
        .sy = 1
    };
}

struct locate_info_t
{
    Cubiomes *cubiomes;
    Generator *g;
    Range r;
    int tol;
    volatile char *stop;
};

static int is_visited(const Cubiomes *cubiomes, size_t index)
{
    return (cubiomes->visited[index >> 3] >> (index & 7)) & 1;
}

static void mark_visited(Cubiomes *cubiomes, size_t index)
{
    cubiomes->visited[index >> 3] |= (uint8_t)(1u << (index & 7));
}

static
int floodFillGen(struct locate_info_t *info, int i, int j, Pos *p)
{
    Cubiomes *cubiomes = info->cubiomes;
    ensure_capacity((void **)&cubiomes->queue, &cubiomes->queue_capacity,
                    1, sizeof(*cubiomes->queue), "flood-fill queue");
    size_t qn = 1;
    cubiomes->queue[0] = (FloodEntry){i, j, 0};
    int64_t sumx = 0;
    int64_t sumz = 0;
    int n = 0;
    while (qn != 0)
    {
        FloodEntry entry = cubiomes->queue[--qn];
        if (info->stop && *info->stop)
            return 0;
        int d = entry.d;
        i = entry.i;
        j = entry.j;
        size_t index = (size_t)j * info->r.sx + i;
        if (is_visited(cubiomes, index))
            continue;
        mark_visited(cubiomes, index);
        int x = info->r.x + i;
        int z = info->r.z + j;
        if (is_mushroom_fields(info->g, info->r.scale, x, info->r.y, z))
        {
            sumx += x;
            sumz += z;
            n++;
            d = 0;
        }
        else
        {
            if (++d >= info->tol)
                continue;
        }
        FloodEntry next[] = { {i,j-1,d}, {i,j+1,d}, {i-1,j,d}, {i+1,j,d} };
        ensure_capacity((void **)&cubiomes->queue, &cubiomes->queue_capacity,
                        qn + 4, sizeof(*cubiomes->queue), "flood-fill queue");
        for (int k = 0; k < 4; k++)
        {
            i = next[k].i; j = next[k].j;
            if (i < 0 || i >= info->r.sx || j < 0 || j >= info->r.sz)
                continue;
            if (is_visited(cubiomes, (size_t)j * info->r.sx + i))
                continue;
            cubiomes->queue[qn++] = next[k];
        }
    }
    if (n)
    {
        p->x = (int) round((sumx / (double)n + 0.5) * info->r.scale);
        p->z = (int) round((sumz / (double)n + 0.5) * info->r.scale);
    }
    return n;
}

static
int getBiomeCentersOpt(Pos *pos, int *siz, int nmax, Cubiomes *cubiomes, Range r,
    int match, int minsiz, int tol, int step, volatile char *stop)
{
    Generator *g = &cubiomes->g;
    if (minsiz <= 0)
        minsiz = 1;
    int i, j, k, n = 0;
    size_t cell_count = (size_t)r.sx * r.sz;
    if (cell_count > UINT32_MAX) {
        fprintf(stderr, "biome-center range is too large\n");
        abort();
    }
    size_t visited_size = (cell_count + 7) / 8;
    ensure_capacity((void **)&cubiomes->visited, &cubiomes->visited_capacity,
                    visited_size, sizeof(*cubiomes->visited), "visited bitmap");
    memset(cubiomes->visited, 0, visited_size);
    if (tol <= 0)
        tol = 1;
    if (step <= 0)
        step = 1;
    struct locate_info_t info;
    info.cubiomes = cubiomes;
    info.g = g;
    info.r = r;
    info.stop = stop;
    info.tol = tol;

    const int *lim = getBiomeParaLimits(g->mc, match);
    size_t candidates_len = 0;

    int para[] = {
        NP_TEMPERATURE,
        NP_HUMIDITY,
        NP_EROSION,
        NP_CONTINENTALNESS,
        NP_WEIRDNESS,
    };
    int npara = sizeof(para) / sizeof(para[0]);
    if (step == 1)
        step = 1 + floor(sqrt(minsiz) * 0.5);

    for (j = 0; j < r.sz; j += step)
    {
        for (i = 0; i < r.sx; i += step)
        {
            if (stop && *stop)
                break;
            int candidate = 1;
            for (k = 0; k < npara; k++)
            {
                const int *plim = lim + 2*para[k];
                if (plim[0] == INT_MIN && plim[1] == INT_MAX)
                    continue;
                DoublePerlinNoise *dpn = &g->bn.climate[para[k]];
                double px = (r.x+i) * r.scale / 4.0;
                double pz = (r.z+j) * r.scale / 4.0;
                int p = 10000 * sampleDoublePerlin(dpn, px, 0, pz);
                if (p < plim[0] || p > plim[1])
                {
                    candidate = 0;
                    break;
                }
            }
            if (candidate) {
                ensure_capacity((void **)&cubiomes->candidates,
                                &cubiomes->candidates_capacity,
                                candidates_len + 1, sizeof(*cubiomes->candidates),
                                "biome-center candidates");
                cubiomes->candidates[candidates_len++] = (uint32_t)((size_t)j * r.sx + i);
            }
        }
    }

    for (size_t candidate_index = 0; candidate_index < candidates_len; candidate_index++)
    {
        size_t index = cubiomes->candidates[candidate_index];
        if (is_visited(cubiomes, index))
            continue;
        i = (int)(index % r.sx);
        j = (int)(index / r.sx);
        Pos center;
        int area = floodFillGen(&info, i, j, &center);
        if (area >= minsiz)
        {
            pos[n] = center;
            if (siz) siz[n] = area;
            if (++n >= nmax)
                break;
        }
    }

    return n;
}

int cubiomes_test_monte_carlo(Cubiomes *cubiomes, int32_t x, int32_t z, int32_t range, int32_t min_area, double confidence) {
    Range r = make_range(x, z, range, 4);
    double fraction = (double)min_area / (r.sx * r.sz * r.scale * r.scale);
    uint64_t rng = cubiomes->g.seed;
    return monteCarloBiomes(&cubiomes->g, r, &rng, fraction, confidence, eval, NULL);
}

int cubiomes_test_biome_centers(Cubiomes *cubiomes, int32_t x, int32_t z, int32_t range, int32_t min_area, int32_t scale, int32_t tol, PosArea *out) {
    Pos pos;
    int siz;
    Range r = make_range(x, z, range, scale);
    int minsiz = min_area / (scale * scale);
    int n = getBiomeCentersOpt(&pos, &siz, 1, cubiomes, r, mushroom_fields, minsiz, tol, 0, NULL);
    if (n == 1) {
        if (out) {
            *out = (PosArea){
                .x = pos.x,
                .z = pos.z,
                .area = siz * (scale * scale),
            };
        }
        return 1;
    }
    return 0;
}


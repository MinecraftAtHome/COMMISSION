#include "cubiomes.h"

#include "../cubiomes/finders.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

struct Cubiomes {
    Generator g;
};

Cubiomes *cubiomes_create(int large_biomes) {
    Cubiomes *cubiomes = malloc(sizeof(Cubiomes));
    if (cubiomes == NULL) {
        fprintf(stderr, "cubiomes_create failed\n");
        abort();
    }
    setupGenerator(&cubiomes->g, MC_NEWEST, large_biomes ? LARGE_BIOMES : 0);
    return cubiomes;
}

void cubiomes_free(Cubiomes *cubiomes) {
    free(cubiomes);
}

void cubiomes_apply_seed(Cubiomes *cubiomes, uint64_t seed) {
    applySeed(&cubiomes->g, DIM_OVERWORLD, seed);
}

static int eval(Generator *g, int scale, int x, int y, int z, void *data) {
    double px = x, pz = z;
    px += sampleDoublePerlin(&g->bn.climate[NP_SHIFT], x, 0, z) * 4.0;
    pz += sampleDoublePerlin(&g->bn.climate[NP_SHIFT], x, z, 0) * 4.0;

    double c = sampleDoublePerlin(&g->bn.climate[NP_CONTINENTALNESS], px, 0, pz);
    return c < -1.05f;
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
    Generator *g;
    int *ids;
    Range r;
    int match, tol;
    volatile char *stop;
};

static
int floodFillGen(struct locate_info_t *info, int i, int j, Pos *p)
{
    typedef struct { int i, j, d; } entry_t;
    entry_t *queue = (entry_t*) malloc(info->r.sx*info->r.sz * sizeof(*queue));
    int qn = 1;
    queue->i = i;
    queue->j = j;
    queue->d = 0;
    int64_t sumx = 0;
    int64_t sumz = 0;
    int n = 0;
    while (--qn >= 0)
    {
        if (info->stop && *info->stop)
        {
            free(queue);
            return 0;
        }
        int d = queue[qn].d;
        i = queue[qn].i;
        j = queue[qn].j;
        int k = j * info->r.sx + i;
        int id = info->ids[k];
        if (id == INT_MAX)
            continue;
        info->ids[k] = INT_MAX;
        int x = info->r.x + i;
        int z = info->r.z + j;
        if (info->g->mc >= MC_1_18)
            id = getBiomeAt(info->g, info->r.scale, x, info->r.y, z);
        if (id == info->match)
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
        entry_t next[] = { {i,j-1,d}, {i,j+1,d}, {i-1,j,d}, {i+1,j,d} };
        for (k = 0; k < 4; k++)
        {
            i = next[k].i; j = next[k].j;
            if (i < 0 || i >= info->r.sx || j < 0 || j >= info->r.sz)
                continue;
            if (info->ids[j * info->r.sx + i] == INT_MAX)
                continue;
            queue[qn++] = next[k];
        }
    }
    free(queue);
    if (n)
    {
        p->x = (int) round((sumx / (double)n + 0.5) * info->r.scale);
        p->z = (int) round((sumz / (double)n + 0.5) * info->r.scale);
    }
    return n;
}

static
int getBiomeCentersOpt(Pos *pos, int *siz, int nmax, Generator *g, Range r,
    int match, int minsiz, int tol, int step, volatile char *stop)
{
    if (minsiz <= 0)
        minsiz = 1;
    int i, j, k, n = 0;
    int *ids = (int*) malloc(r.sx*r.sz * sizeof(int));
    memset(ids, -1, r.sx*r.sz * sizeof(int));
    if (tol <= 0)
        tol = 1;
    if (step <= 0)
        step = 1;
    struct locate_info_t info;
    info.g = g;
    info.ids = ids;
    info.r = r;
    info.stop = stop;
    info.match = match;
    info.tol = tol;

    if (g->mc >= MC_1_18)
    {
        const int *lim = getBiomeParaLimits(g->mc, match);

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
                        ids[j*r.sx + i] = -2;
                        break;
                    }
                }
            }
        }
        match = -1; // id entries that are still -1 are our candidates
    }
    else // 1.17-
    {
        int ts = 32 / r.scale;
        if (r.sx + r.sz < 32)
            ts = 8;

        int tx = (int) floor(r.x / (double)ts);
        int tz = (int) floor(r.z / (double)ts);
        int tw = (int) ceil((r.x+r.sx) / (double)ts) - tx;
        int th = (int) ceil((r.z+r.sz) / (double)ts) - tz;
        int ti, tj;

        BiomeFilter bf;
        setupBiomeFilter(&bf, g->mc, 0, &match, 1, 0, 0, 0, 0);
        //applySeed(g, 0, g->seed);

        Range tr = { r.scale, 0, 0, ts, ts, 0, 1 };
        int *cache = allocCache(g, r);

        for (tj = 0; tj < th; tj++)
        {
            for (ti = 0; ti < tw; ti++)
            {
                if (stop && *stop)
                    break;
                tr.x = (tx+ti) * ts;
                tr.z = (tz+tj) * ts;
                if (checkForBiomes(g, cache, tr, DIM_OVERWORLD, g->seed,
                    &bf, stop) != 1)
                {
                    continue;
                }
                for (j = 0; j < ts; j++)
                {
                    int jj = tr.z + j - r.z;
                    if (jj < 0 || jj >= r.sz)
                        continue;
                    for (i = 0; i < ts; i++)
                    {
                        int ii = tr.x + i - r.x;
                        if (ii < 0 || ii >= r.sx)
                            continue;
                        ids[jj*r.sx + ii] = cache[j*tr.sx + i];
                    }
                }
            }
        }
        free(cache);
    }

    // applySeed(g, DIM_OVERWORLD, g->seed);
    for (j = 0; j < r.sz; j += step)
    {
        for (i = 0; i < r.sx; i += step)
        {
            if (stop && *stop)
                break;
            if (ids[j*r.sx + i] != match)
                continue;
            Pos center;
            int area = floodFillGen(&info, i, j, &center);
            if (area >= minsiz)
            {
                pos[n] = center;
                if (siz) siz[n] = area;
                if (++n >= nmax)
                    goto L_end;
            }
        }
    }

L_end:
    free(ids);

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
    int n = getBiomeCentersOpt(&pos, &siz, 1, &cubiomes->g, r, mushroom_fields, minsiz, tol, 0, NULL);
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

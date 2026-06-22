#include <cstdint>

#ifndef HOST_DEVICE
#ifdef __CUDA_ARCH__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#endif

template<typename R>
struct WorldgenRandom {
    using BaseRandom = R;

    BaseRandom base_random;

    HOST_DEVICE explicit WorldgenRandom() : base_random() {

    }

    HOST_DEVICE explicit WorldgenRandom(const BaseRandom &base_random) : base_random(base_random) {

    }

    HOST_DEVICE explicit WorldgenRandom(uint64_t seed) : base_random() {
        this->setSeed(seed);
    }

    HOST_DEVICE WorldgenRandom(const WorldgenRandom &other) : base_random(other.base_random) {

    }

    HOST_DEVICE void setSeed(uint64_t seed) {
        this->base_random.setSeed(seed);
    }

    HOST_DEVICE uint32_t nextBits(int32_t bits) {
        return this->base_random.nextBits(bits);
    }

    HOST_DEVICE uint32_t nextInt() {
        return static_cast<uint32_t>(this->nextBits(32));
    }

    HOST_DEVICE uint32_t nextInt(uint32_t bound) {
        uint32_t r = this->nextBits(31);
        uint32_t m = bound - 1;
        if ((bound & m) == 0) {
            r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
        } else {
            for (uint32_t u = r;
                    static_cast<int32_t>(u - (r = u % bound) + m) < 0;
                    u = this->nextBits(31))
                ;
        }
        return r;
    }

    HOST_DEVICE uint32_t nextIntFast(uint32_t bound) {
        uint32_t r = this->nextBits(31);
        uint32_t m = bound - 1;
        if ((bound & m) == 0) {
            r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
        } else {
            r = r % bound;
        }
        return r;
    }

    HOST_DEVICE uint64_t nextLong() {
        return (static_cast<uint64_t>(this->nextBits(32)) << 32) + static_cast<uint64_t>(static_cast<int32_t>(this->nextBits(32)));
    }

    HOST_DEVICE bool nextBoolean() {
        return this->nextBits(1) != 0;
    }

    HOST_DEVICE float nextFloat() {
        return static_cast<float>(this->nextBits(24)) * 0x1.0p-24f;
    }

    HOST_DEVICE double nextDouble() {
        uint32_t a = this->nextBits(26);
        uint32_t b = this->nextBits(27);
        return static_cast<double>((static_cast<uint64_t>(a) << 27) + static_cast<uint64_t>(b)) * 0x1.0p-53;
    }

    HOST_DEVICE uint64_t getDecorationSeed12(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
        this->setSeed(world_seed);
        uint64_t a = static_cast<int64_t>(this->nextLong()) / 2 * 2 + 1;
        uint64_t b = static_cast<int64_t>(this->nextLong()) / 2 * 2 + 1;
        return (static_cast<uint64_t>(chunk_x) * a + static_cast<uint64_t>(chunk_z) * b) ^ world_seed;
    }

    HOST_DEVICE uint64_t getDecorationSeed13(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
        this->setSeed(world_seed);
        uint64_t a = this->nextLong() | 1;
        uint64_t b = this->nextLong() | 1;
        return (static_cast<uint64_t>(chunk_x * 16) * a + static_cast<uint64_t>(chunk_z * 16) * b) ^ world_seed;
    }

    HOST_DEVICE void setFeatureSeed(uint64_t decoration_seed, uint32_t salt) {
        this->setSeed(decoration_seed + static_cast<uint64_t>(salt));
    }

    HOST_DEVICE void setFeatureSeed12(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z, uint32_t salt) {
        this->setFeatureSeed(this->getDecorationSeed12(world_seed, chunk_x, chunk_z), salt);
    }

    HOST_DEVICE void setFeatureSeed13(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z, uint32_t salt) {
        this->setFeatureSeed(this->getDecorationSeed13(world_seed, chunk_x, chunk_z), salt);
    }

    HOST_DEVICE void setLargeFeatureSeed(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
        this->setSeed(world_seed);
        uint64_t a = this->nextLong();
        uint64_t b = this->nextLong();
        this->setSeed((static_cast<uint64_t>(chunk_x) * a ^ static_cast<uint64_t>(chunk_z) * b) ^ world_seed);
    }

    HOST_DEVICE void setLargeFeatureWithSalt(uint64_t world_seed, int32_t region_x, int32_t region_z, int32_t salt) {
        this->setSeed(static_cast<uint64_t>(region_x) * UINT64_C(341873128712) + static_cast<uint64_t>(region_z) * UINT64_C(132897987541) + world_seed + static_cast<uint64_t>(salt));
    }
};

struct JavaRandomSkip {
    static constexpr uint64_t MULTIPLIER = UINT64_C(0x5DEECE66D);
    static constexpr uint64_t ADDEND = UINT64_C(0xB);
    static constexpr uint64_t MASK = UINT64_C(0xFFFFFFFFFFFF);

    uint64_t multiplier;
    uint64_t addend;

    HOST_DEVICE constexpr explicit JavaRandomSkip(uint64_t multiplier, uint64_t addend) : multiplier(multiplier), addend(addend) {

    }

    HOST_DEVICE constexpr void combine(const JavaRandomSkip &other) {
        uint64_t combined_multiplier = (multiplier * other.multiplier) & MASK;
        uint64_t combined_addend = (addend * other.multiplier + other.addend) & MASK;
        multiplier = combined_multiplier;
        addend = combined_addend;
    }

    HOST_DEVICE static constexpr JavaRandomSkip make(int64_t n) {
        uint64_t n_mod = static_cast<uint64_t>(n) & MASK;

        JavaRandomSkip skip(1, 0);
        JavaRandomSkip skip_pow(MULTIPLIER, ADDEND);

        for (uint64_t i = 0; i < 48; i++) {
            if (n_mod & (UINT64_C(1) << i)) {
                skip.combine(skip_pow);
            }

            skip_pow.combine(skip_pow);
        }

        return skip;
    }
};

struct JavaRandom {
    uint64_t seed;

    HOST_DEVICE explicit JavaRandom() {

    }

    HOST_DEVICE explicit JavaRandom(uint64_t seed) {
        this->setSeed(seed);
    }

    HOST_DEVICE JavaRandom(const JavaRandom &other) : seed(other.seed) {

    }

    HOST_DEVICE static JavaRandom withSeed(uint64_t seed) {
        JavaRandom random;
        random.seed = seed;
        return random;
    }

    HOST_DEVICE void setSeed(uint64_t seed) {
        this->seed = (seed ^ JavaRandomSkip::MULTIPLIER) & JavaRandomSkip::MASK;
    }

    template<int64_t N = 1>
    HOST_DEVICE JavaRandom &skip() {
        constexpr JavaRandomSkip skip = JavaRandomSkip::make(N);
        this->seed = (this->seed * skip.multiplier + skip.addend) & JavaRandomSkip::MASK;
        return *this;
    }

    template<int64_t N = 1>
    HOST_DEVICE uint32_t nextBits(int32_t bits) {
        this->skip<N>();
        return static_cast<uint32_t>(this->seed >> (48 - bits));
    }

    template<int64_t N = 1>
    HOST_DEVICE uint32_t nextInt() {
        return this->nextBits<N>(32);
    }

    template<int64_t N = 1>
    HOST_DEVICE uint32_t nextInt(uint32_t bound) {
        uint32_t r = this->nextBits<N>(31);
        uint32_t m = bound - 1;
        if ((bound & m) == 0) {
            r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
        } else {
            for (uint32_t u = r;
                    static_cast<int32_t>(u - (r = u % bound) + m) < 0;
                    u = this->nextBits<1>(31))
                ;
        }
        return r;
    }

    template<int64_t N = 1>
    HOST_DEVICE uint32_t nextIntFast(uint32_t bound) {
        uint32_t r = this->nextBits<N>(31);
        uint32_t m = bound - 1;
        if ((bound & m) == 0) {
            r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
        } else {
            r = r % bound;
        }
        return r;
    }

    template<int64_t N = 1>
    HOST_DEVICE uint64_t nextLong() {
        uint32_t a = this->nextBits<N>(32);
        uint32_t b = this->nextBits<1>(32);
        return (static_cast<uint64_t>(a) << 32) + static_cast<uint64_t>(static_cast<int32_t>(b));
    }

    template<int64_t N = 1>
    HOST_DEVICE bool nextBoolean() {
        return this->nextBits<N>(1) != 0;
    }

    template<int64_t N = 1>
    HOST_DEVICE float nextFloat() {
        return static_cast<float>(this->nextBits<N>(24)) * 0x1.0p-24f;
    }

    template<int64_t N = 1>
    HOST_DEVICE double nextDouble() {
        uint32_t a = this->nextBits<N>(26);
        uint32_t b = this->nextBits<1>(27);
        return static_cast<double>((static_cast<uint64_t>(a) << 27) + static_cast<uint64_t>(b)) * 0x1.0p-53;
    }
};

struct XrsrForkHash {
    uint64_t lo;
    uint64_t hi;
};

struct XrsrRandom;

struct XrsrRandomFork {
    uint64_t lo;
    uint64_t hi;

    HOST_DEVICE XrsrRandom from(const XrsrForkHash &hash) const;
};

struct XrsrRandom {
    constexpr static uint64_t XRSR_MIX1 = 0xbf58476d1ce4e5b9;
    constexpr static uint64_t XRSR_MIX2 = 0x94d049bb133111eb;
    constexpr static uint64_t XRSR_MIX1_INVERSE = 0x96de1b173f119089;
    constexpr static uint64_t XRSR_MIX2_INVERSE = 0x319642b2d24d8ec3;
    constexpr static uint64_t XRSR_SILVER_RATIO = 0x6a09e667f3bcc909;
    constexpr static uint64_t XRSR_GOLDEN_RATIO = 0x9e3779b97f4a7c15;

    HOST_DEVICE static uint64_t mix64(uint64_t a) {
        a = (a ^ a >> 30) * XRSR_MIX1;
        a = (a ^ a >> 27) * XRSR_MIX2;
        return a ^ a >> 31;
    }

    HOST_DEVICE static uint64_t fix64(uint64_t a) {
        a = (a ^ a >> 31 ^ a >> 62) * XRSR_MIX2_INVERSE;
        a = (a ^ a >> 27 ^ a >> 54) * XRSR_MIX1_INVERSE;
        return a ^ a >> 30 ^ a >> 60;
    }

    HOST_DEVICE constexpr static uint64_t rol64(uint64_t a, int bits) {
        return (a << bits) | (a >> (64 - bits));
    }

    uint64_t lo;
    uint64_t hi;

    HOST_DEVICE explicit XrsrRandom() {

    }

    HOST_DEVICE constexpr XrsrRandom(uint64_t lo, uint64_t hi) : lo(lo), hi(hi) {
        if ((this->lo | this->hi) == 0) {
            this->lo = static_cast<uint64_t>(-7046029254386353131);
            this->hi = static_cast<uint64_t>(7640891576956012809);
        }
    }

    HOST_DEVICE explicit XrsrRandom(uint64_t seed) {
        this->setSeed(seed);
    }

    HOST_DEVICE XrsrRandom(const XrsrRandom &other) : lo(other.lo), hi(other.hi) {

    }

    HOST_DEVICE void setSeed(uint64_t seed) {
        seed ^= XRSR_SILVER_RATIO;
        this->lo = mix64(seed);
        this->hi = mix64(seed + XRSR_GOLDEN_RATIO);

        if ((this->lo | this->hi) == 0) {
            this->lo = static_cast<uint64_t>(-7046029254386353131);
            this->hi = static_cast<uint64_t>(7640891576956012809);
        }
    }

    HOST_DEVICE constexpr uint64_t nextInternal() {
        uint64_t l = this->lo;
        uint64_t h = this->hi;
        uint64_t r = rol64(l + h, 17) + l;
        h ^= l;
        this->lo = rol64(l, 49) ^ h ^ h << 21;
        this->hi = rol64(h, 28);
        return r;
    }

    HOST_DEVICE uint64_t nextBits(int32_t bits) {
        return this->nextInternal() >> (64 - bits);
    }

    HOST_DEVICE uint32_t nextInt() {
        return static_cast<uint32_t>(this->nextInternal());
    }

    HOST_DEVICE uint32_t nextInt(uint32_t bound) {
        uint64_t l = this->nextInt();
        uint64_t m = l * bound;
        uint64_t n = m & 0xFFFFFFFF;
        if (n < bound) {
            uint32_t j = (~bound + 1) % bound;
            while (n < j) {
                l = this->nextInt();
                m = l * bound;
                n = m & 0xFFFFFFFFL;
            }
        }
        uint64_t o = m >> 32;
        return o;
    }

    HOST_DEVICE uint32_t nextIntFast(uint32_t bound) {
        uint64_t l = this->nextInt();
        uint64_t m = l * bound;
        uint64_t o = m >> 32;
        return o;
    }

    HOST_DEVICE uint64_t nextLong() {
        return this->nextInternal();
    }

    HOST_DEVICE bool nextBoolean() {
        return (this->nextInternal() & 1) != 0;
    }

    HOST_DEVICE float nextFloat() {
        return static_cast<float>(this->nextBits(24)) * 5.9604645E-8f;
    }

    HOST_DEVICE double nextDouble() {
        return static_cast<double>(this->nextBits(53)) * 1.110223E-16f;
    }

    HOST_DEVICE XrsrRandomFork fork() {
        uint64_t lo = this->nextLong();
        uint64_t hi = this->nextLong();
        return { lo, hi };
    }
};

HOST_DEVICE XrsrRandom XrsrRandomFork::from(const XrsrForkHash &hash) const {
    return { lo ^ hash.lo, hi ^ hash.hi };
}
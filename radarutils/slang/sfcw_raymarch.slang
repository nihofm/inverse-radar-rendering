// ---------------------------------------------------
// Constants

static const float M_PI = 3.14159265358979323846; // pi

// ---------------------------------------------------
// Random generator helper

// tiny encryption algorithm (TEA) to calculate a seed per launch index and iteration
uint tea(const uint val0, const uint val1, const uint N) {
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;
    for (uint n = 0; n < N; ++n) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

// return a random sample in the range [0, 1) with a simple linear congruential generator
float rng(inout uint previous) {
    previous = previous * 1664525u + 1013904223u;
    return float(previous & 0x00FFFFFFu) / float(0x01000000u);
}

float3 rng3(inout uint previous) {
    return float3(rng(previous), rng(previous), rng(previous));
}

// -----------------------------------------------
// Intersection helpers

bool intersect_box(const float3 pos, const float3 dir, const float3 bb_min, const float3 bb_max, out float2 near_far) {
    const float3 inv_dir = 1.f / dir;
    const float3 lo = (bb_min - pos) * inv_dir;
    const float3 hi = (bb_max - pos) * inv_dir;
    const float3 tmin = min(lo, hi), tmax = max(lo, hi);
    near_far.x = max(0.f, max(tmin.x, max(tmin.y, tmin.z)));
    near_far.y = min(tmax.x, min(tmax.y, tmax.z));
    return near_far.x <= near_far.y;
}

// -----------------------------------------------
// Reco raymarch kernel for perspective view

[AutoPyBindCUDA]
[CudaKernel]
void sfcw_raymarch_kernel(
    TensorView<float3> cam_pos,
    TensorView<float3> cam_dir,
    TensorView<float> cam_fovy,
    TensorView<float3> AABB,
    TensorView<float> reco,
    const int seed,
    TensorView<float2> result
) {
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx(); // data layout: [n_batch, height, width]
    if (idx.x >= result.size(0) || idx.y >= result.size(1) || idx.z >= result.size(2)) return;
    // setup RNG
    uint rng_state = tea(idx.y * result.size(2) + idx.z, seed, 32);
    // setup view ray
    const float2 pixel = float2(idx.z + rng(rng_state), idx.y + rng(rng_state));
    const float2 xy = (pixel - float2(result.size(2), result.size(1)) * 0.5f) / float(result.size(1));
    const float z = 0.5f / tan(0.5f * M_PI * cam_fovy[idx.x] / 180.f);
    // compute coordinate frame
    const float3 ndir = normalize(cam_dir[idx.x]);
    const float3 left = normalize(cross(ndir, float3(1e-7f, 1.f, 1e-6f)));
    const float3 up = cross(-left, ndir);
    // transform to world space
    const float3 pos = cam_pos[idx.x];
    const float3 dir = normalize(xy.x * left + xy.y * up + z * ndir);
    // intersect scene AABB
    float2 near_far;
    const bool hit = intersect_box(pos, dir, AABB[0], AABB[1], near_far);
    const uint3 n_voxels = uint3(reco.size(0), reco.size(1), reco.size(2));
    // raymarch
    float2 reco_t = float2(0); // (max_reco, depth)
    if (hit) {
        static const int RAYMARCH_ITER = 128;
        const float dt = (near_far.y - near_far.x) / float(RAYMARCH_ITER);
        near_far.x += rng(rng_state) * dt; // jitter starting position
        for (int i = 0; i < RAYMARCH_ITER; ++i) {
            const float t = min(near_far.x + i * dt, near_far.y);
            const float3 reco_pos = pos + t * dir;
            const float3 iposf = n_voxels * (reco_pos - AABB[0]) / (AABB[1] - AABB[0]);
            const uint3 ipos = clamp(uint3(iposf - .5f + rng3(rng_state)), uint3(0), n_voxels - 1);
            const float intensity = reco[ipos];
            if (intensity > reco_t.x)
                reco_t = float2(intensity, t);
        }
    }
    result[idx] = reco_t;
}

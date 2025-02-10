// ---------------------------------------------------
// constants

static const float M_PI = 3.14159265358979323846;

// ---------------------------------------------------
// random generator helper

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

float3 spherical_to_cartesian(float theta, float phi) {
    return float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
}

// ---------------------------------------------------
// pinhole camera

[AutoPyBindCUDA]
[CudaKernel]
void raygen_pinhole_camera(
    TensorView<float3> cam_pos,
    TensorView<float3> cam_dir,
    TensorView<float> cam_fovy,
    TensorView<float3> out_pos,
    TensorView<float3> out_dir)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx(); // launch layout: [batch, height, width]
    if (idx.x >= out_pos.size(0) || idx.y >= out_pos.size(1) || idx.z >= out_pos.size(2)) return;
    const float2 pixel = float2(idx.z + 0.5f, idx.y + 0.5f); // TODO: SSAA
    const float2 xy = (pixel - float2(out_pos.size(2), out_pos.size(1)) * 0.5f) / float(out_pos.size(1));
    const float z = 0.5f / tan(0.5f * M_PI * cam_fovy[idx.x] / 180.f);
    // compute coordinate frame
    const float3 ndir = normalize(cam_dir[idx.x]);
    const float3 left = normalize(cross(ndir, float3(1e-7f, 1.f, 1e-6f)));
    const float3 up = cross(-left, ndir);
    // transform to world space
    out_pos[idx] = cam_pos[idx.x];
    out_dir[idx] = normalize(xy.x * left + xy.y * up + z * ndir);
}

// ---------------------------------------------------
// random rays

[AutoPyBindCUDA]
[CudaKernel]
void raygen_random_aabb(
    TensorView<float3> origins,
    TensorView<float3> AABB,
    const uint seed,
    TensorView<float3> out_pos,
    TensorView<float3> out_dir)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx(); // launch layout: [batch, height, width]
    if (idx.x >= out_pos.size(0) || idx.y >= out_pos.size(1) || idx.z >= out_pos.size(2)) return;
    const float3 pos = origins[idx.x]; // one origin per batch
    uint rng_state = tea(idx.y * out_pos.size(2) + idx.z, seed + idx.x, 32);
    // sample position in AABB
    const float3 tc = float3(rng(rng_state), rng(rng_state), rng(rng_state));
    const float3 target = AABB[0] + tc * (AABB[1] - AABB[0]);
    out_pos[idx] = pos;
    out_dir[idx] = normalize(target - pos);
}

[AutoPyBindCUDA]
[CudaKernel]
void raygen_uniform_hemisphere(
    TensorView<float3> origins,
    TensorView<float3> normals,
    const uint seed,
    TensorView<float3> out_pos,
    TensorView<float3> out_dir)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx(); // launch layout: [batch, height, width]
    if (idx.x >= out_pos.size(0) || idx.y >= out_pos.size(1) || idx.z >= out_pos.size(2)) return;
    const float3 pos = origins[idx.x]; // one origin per batch
    const float3 dir = normals[idx.x]; // one normal per batch
    uint rng_state = tea(idx.y * out_pos.size(2) + idx.z, seed + idx.x, 32);
    // uniform sample hemisphere
    const float z = rng(rng_state);
    const float r = sqrt(max(0, 1 - z * z));
    const float phi = 2 * M_PI * rng(rng_state);
    const float3 sample = float3(r * cos(phi), r * sin(phi), z);
    // compute coordinate frame
    const float3 ndir = normalize(dir);
    const float3 left = normalize(cross(ndir, float3(1e-7f, 1.f, 1e-6f)));
    const float3 up = cross(-left, ndir);
    // transform to world space
    out_pos[idx] = pos;
    out_dir[idx] = normalize(sample.x * left + sample.y * up + sample.z * ndir);
}

[AutoPyBindCUDA]
[CudaKernel]
void raygen_cosine_hemisphere(
    TensorView<float3> origins,
    TensorView<float3> normals,
    const uint seed,
    TensorView<float3> out_pos,
    TensorView<float3> out_dir)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx(); // launch layout: [batch, height, width]
    if (idx.x >= out_pos.size(0) || idx.y >= out_pos.size(1) || idx.z >= out_pos.size(2)) return;
    const float3 pos = origins[idx.x]; // one origin per batch
    const float3 dir = normals[idx.x]; // one normal per batch
    uint rng_state = tea(idx.y * out_pos.size(2) + idx.z, seed + idx.x, 32);
    // uniform sample disk
    const float r = sqrt(rng(rng_state));
    const float theta = 2 * M_PI * rng(rng_state);
    const float2 disk = r * float2(cos(theta), sin(theta));
    // cosine sample hemisphere
    const float z = 1 - disk.x * disk.x - disk.y * disk.y;
    const float3 sample = float3(disk.x, disk.y, z > 0 ? sqrt(z) : 0.f);
    // compute coordinate frame
    const float3 ndir = normalize(dir);
    const float3 left = normalize(cross(ndir, float3(1e-7f, 1.f, 1e-6f)));
    const float3 up = cross(-left, ndir);
    // transform to world space
    out_pos[idx] = pos;
    out_dir[idx] = normalize(sample.x * left + sample.y * up + sample.z * ndir);
}

[AutoPyBindCUDA]
[CudaKernel]
void raygen_cosine_power_hemisphere(
    TensorView<float3> origins,
    TensorView<float3> normals,
    TensorView<float> power,
    const uint seed,
    TensorView<float3> out_pos,
    TensorView<float3> out_dir)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx(); // launch layout: [batch, height, width]
    if (idx.x >= out_pos.size(0) || idx.y >= out_pos.size(1) || idx.z >= out_pos.size(2)) return;
    const float3 pos = origins[idx.x]; // one origin per batch
    const float3 dir = normals[idx.x]; // one normal per batch
    uint rng_state = tea(idx.y * out_pos.size(2) + idx.z, seed + idx.x, 32);
    // cosine power sampling
    const float theta = acos(pow(rng(rng_state), 1 / (power[idx.x] + 1)));
    const float phi = 2 * M_PI * rng(rng_state);
    const float3 sample = spherical_to_cartesian(theta, phi);
    // compute coordinate frame
    const float3 ndir = normalize(dir);
    const float3 left = normalize(cross(ndir, float3(1e-7f, 1.f, 1e-6f)));
    const float3 up = cross(-left, ndir);
    // transform to world space
    out_pos[idx] = pos;
    out_dir[idx] = normalize(sample.x * left + sample.y * up + sample.z * ndir);
}

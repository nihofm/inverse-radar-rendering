#include "params.h"

extern "C" {
	__constant__ HitgenParams params;
}

// math helpers
static const float PI = 3.14159265358979323846;
inline __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __device__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __device__ float3 operator*(float b, float3 a) { return make_float3(b * a.x, b * a.y, b * a.z); }
inline __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ float3 normalize(float3 v) { return v * rsqrtf(dot(v, v)); }
inline __device__ float3 cross(float3 a, float3 b) { return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }

__device__ float3 spherical_to_cartesian(float theta, float phi) {
    return make_float3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
}

// tiny encryption algorithm (TEA) to calculate a seed per launch index and iteration
__device__ unsigned int tea(const unsigned int val0, const unsigned int val1, const unsigned int N) {
	unsigned int v0 = val0; 
	unsigned int v1 = val1;
	unsigned int s0 = 0;
	for (unsigned int n = 0; n < N; ++n) {
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
		v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
	}
	return v0;
}

// return a random sample in the range [0, 1) with a simple linear congruential generator
__device__ float rng(unsigned int& previous) {
	previous = previous * 1664525u + 1013904223u;
	return float(previous & 0x00FFFFFFu) / float(0x01000000u);
}

extern "C" __global__ void __raygen__rg() {
	const unsigned int idx = optixGetLaunchIndex().x;
	unsigned int rng_state = tea(idx, params.seed, 32);
	while (true) {
		// sample origin
		const int org_idx = floor(rng(rng_state) * params.n_orgs);
		// cosine power sampling
		const float theta = acos(pow(rng(rng_state), 1 / (params.power + 1)));
		const float phi = 2 * PI * rng(rng_state);
		const float3 sample = spherical_to_cartesian(theta, phi);
		// compute coordinate frame
		const float3 ndir = normalize(params.directions[org_idx]);
		const float3 left = normalize(cross(ndir, make_float3(1e-7f, 1.f, 1e-6f)));
		const float3 up = cross(-1 * left, ndir);
		const float3 sample_dir = normalize(sample.x * left + sample.y * up + sample.z * ndir);
		// trace ray
		unsigned int p0, p1, p2, p3, p4;
		optixTrace(
			params.handle,
			params.origins[org_idx],
			sample_dir,
			0.f,					  // min intersection distance
			1e16f,					  // max intersection distance
			0.f,					  // ray time for motion blur
			OptixVisibilityMask(255), // always visible
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0, // SBT offset
			1, // SBT stride
			0, // missSBTIndex
			p0, p1, p2, p3, p4
		);
		// check and store hits
		if (p0 != 0xFFFFFFFF) {
			int hit_idx = atomicAdd(&params.counter[0], 1);
			if (hit_idx < params.n_hits) {
				params.hit_positions[hit_idx] = params.origins[org_idx] + __uint_as_float(p0) * sample_dir;
				params.hit_normals[hit_idx] = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
				params.hit_primitiveIDs[hit_idx] = p4;
			}
		}
		if (params.counter[0] >= params.n_hits)
			break;
	}
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0(0xFFFFFFFF);
}

extern "C" __global__ void __closesthit__ch() {
	optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
	// lookup vertices
	OptixTraversableHandle gas = optixGetGASTraversableHandle();
	unsigned int primIdx = optixGetPrimitiveIndex();
	unsigned int sbtIdx = optixGetSbtGASIndex();
	float time = optixGetRayTime();
	float3 vertices[3];
	optixGetTriangleVertexData(gas, primIdx, sbtIdx, time, vertices);
	// compute and store normal
	const float3 N = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
	optixSetPayload_1(__float_as_uint(optixIsBackFaceHit() ? -N.x : N.x));
	optixSetPayload_2(__float_as_uint(optixIsBackFaceHit() ? -N.y : N.y));
	optixSetPayload_3(__float_as_uint(optixIsBackFaceHit() ? -N.z : N.z));
	optixSetPayload_4(primIdx);
}

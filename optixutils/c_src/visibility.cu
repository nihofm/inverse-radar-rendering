#include "params.h"

extern "C" {
	__constant__ VisibilityParams params;
}

inline __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __device__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __device__ float3 operator*(float b, float3 a) { return make_float3(b * a.x, b * a.y, b * a.z); }
inline __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ float length(float3 a) { return sqrt(dot(a, a)); }
inline __device__ float3 normalize(float3 v) { return v * rsqrtf(dot(v, v)); }

extern "C" __global__ void __raygen__rg() {
	const unsigned int idx = optixGetLaunchIndex().x;

	const float3 ray_origin = params.ray_origins[idx];
	const float3 ray_target = params.ray_targets[idx];
	const float3 ray_direction = normalize(ray_target - ray_origin);
	const float ray_length = length(ray_target - ray_origin);

	unsigned int p0;
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		1e-4f,                		// min intersection distance
		ray_length - 2e-4f,         // max intersection distance
		0.f,                		// ray time for motion blur
		OptixVisibilityMask(255), 	// always visible
		OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		0,                   		// SBT offset
		1,                   		// SBT stride
		0,                   		// missSBTIndex
		p0
	);

	params.visibility[idx] = __uint_as_float(p0);
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0(__float_as_uint(1.f));
}

extern "C" __global__ void __closesthit__ch() {
	optixSetPayload_0(__float_as_uint(0.f));
}

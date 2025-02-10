#include "params.h"

extern "C" {
	__constant__ RaytraceParams params;
}

inline __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __device__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __device__ float3 operator*(float b, float3 a) { return make_float3(b * a.x, b * a.y, b * a.z); }
inline __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ float3 normalize(float3 v) { return v * rsqrtf(dot(v, v)); }
inline __device__ float3 cross(float3 a, float3 b) { return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }

extern "C" __global__ void __raygen__rg() {
	const unsigned int idx = optixGetLaunchIndex().x;
	const float3 ray_origin = params.ray_origins[idx];
	const float3 ray_direction = params.ray_directions[idx];

	unsigned int p0, p1, p2, p3, p4;
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		1e-4f,                		// min intersection distance
		1e16f,	           			// max intersection distance
		0.f,                		// ray time for motion blur
		OptixVisibilityMask(255), 	// always visible
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,                   		// SBT offset
		1,                   		// SBT stride
		0,                   		// missSBTIndex
		p0, p1, p2, p3, p4
	);

	params.hit_positions[idx] = ray_origin + __uint_as_float(p0) * ray_direction;
	params.hit_normals[idx] = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
	params.hit_primitiveIDs[idx] = p4;
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
	optixSetPayload_1(__float_as_uint(0));
	optixSetPayload_2(__float_as_uint(0));
	optixSetPayload_3(__float_as_uint(0));
	optixSetPayload_4(0xFFFFFFFF);
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
	optixSetPayload_4(optixGetPrimitiveIndex());
}

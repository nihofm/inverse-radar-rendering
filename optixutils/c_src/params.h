#pragma once

#include <optix/optix.h>

struct RaytraceParams {
    // inputs
    OptixTraversableHandle handle;
    const float3* ray_origins;
    const float3* ray_directions;
    // outputs
    float3* hit_positions;
    float3* hit_normals;
    int* hit_primitiveIDs;
};

struct VisibilityParams {
    // inputs
    OptixTraversableHandle handle;
    const float3* ray_origins;
    const float3* ray_targets;
    // outputs
    float* visibility;
};

struct HitgenParams {
    // inputs
    OptixTraversableHandle handle;
    float3* origins;
    float3* directions;
    float power;
    int seed;
    int* counter;
    int n_orgs;
    int n_hits;
    // outputs
    float3* hit_positions;
    float3* hit_normals;
    int* hit_primitiveIDs;
};

#pragma once

#include <filesystem>
#include <optix/optix.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

// ------------------------------------------
// Optix context

struct OptixContext {
    OptixContext();
    virtual ~OptixContext();

    // ---------------------------------
    // bvh construction queries, expects tensors in [..., 3] layout, float3 vertex and int3 index data

    void build_bvh(at::Tensor vertex_buffer);
    void build_bvh(at::Tensor vertex_buffer, at::Tensor index_buffer);

    // ---------------------------------
    // closest hit query for given rays
    // expects float tensors in [..., 3] layout
    // returns hit positions, hit normals and hit primitive IDs (or -1 on miss) as tensors

    std::tuple<at::Tensor, at::Tensor, at::Tensor> raytrace(at::Tensor ray_origins, at::Tensor ray_directions);

    // ---------------------------------
    // visibility query between two points
    // expects float tensors in [..., 3] layout
    // returns visibility, i.e. 0 on occlusion and 1 otherwise, as float tensor

    at::Tensor visibility(at::Tensor ray_origins, at::Tensor ray_target);

    // ---------------------------------
    // ray generation query using cosine power sampling, expects tensors in [N, 3] layout
    // returns compacted tensors containing hit positions [n_hits, 3], hit normals [n_hits, 3] and hit primIDs [n_hits, 1]

    std::tuple<at::Tensor, at::Tensor, at::Tensor> hitgen_cosine_power(at::Tensor origins, at::Tensor directions, float power, uint32_t seed, uint32_t n_hits);

    // data
    OptixDeviceContext context;
    OptixPipeline raytrace_pipeline;
    OptixShaderBindingTable raytrace_sbt;
    OptixPipeline visibility_pipeline;
    OptixShaderBindingTable visibility_sbt;
    OptixPipeline hitgen_pipeline;
    OptixShaderBindingTable hitgen_sbt;
    OptixTraversableHandle accel_handle;
    CUdeviceptr d_gas_output_buffer;
    at::Tensor AABB;
};

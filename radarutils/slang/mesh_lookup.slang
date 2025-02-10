// -----------------------------------------------
// Mesh sample kernel

[Differentiable]
float sigmoid(const float x) {
    return 1 / (1 + exp(-x));
}

[Differentiable]
float triangle_area(float3 A, float3 B, float3 C) {
    return 0.5 * length(cross(B - A, C - A));
}

[Differentiable]
float vertex_lookup(
    no_diff TensorView<float3> hit_positions, // [n_hits, 3]
    no_diff TensorView<int> hit_primIDs,      // [n_hits, 1]
    no_diff TensorView<float3> vbo,           // [n_vertices, 3]
    no_diff TensorView<int3> ibo,             // [n_faces, 3]
    DiffTensorView<float> features,           // [n_vertices, n_features]
    uint2 idx
) {
    const float3 pos = no_diff hit_positions[idx.x];
    const int primID = hit_primIDs[idx.x];
    if (primID < 0 || primID >= ibo.size(0)) return 0.f;
    const int3 face = no_diff ibo[primID];
    const float3 A = no_diff vbo[face.x];
    const float3 B = no_diff vbo[face.y];
    const float3 C = no_diff vbo[face.z];
    // compute baryzentric coords and interpolate features
    const float full = triangle_area(A, B, C);
    const float alpha = triangle_area(pos, B, C) / full;
    const float beta = triangle_area(A, pos, C) / full;
    const float gamma = triangle_area(A, B, pos) / full;
    return
        alpha * sigmoid(features[face.x, idx.y]) +
        beta * sigmoid(features[face.y, idx.y]) +
        gamma * sigmoid(features[face.z, idx.y]);
}

// ---------------------------------------------------
// Forward pass

[CudaKernel]
void mesh_lookup_fwd_kernel(
    TensorView<float3> hit_positions,   // [n_hits, 3]
    TensorView<int> hit_primIDs,        // [n_hits, 1]
    TensorView<float3> vbo,             // [n_vertices, 3]
    TensorView<int3> ibo,               // [n_faces, 3]
    DiffTensorView<float> features,     // [n_vertices, n_features]
    TensorView<float> output
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= hit_positions.size(0) || idx.y >= features.size(1)) return;
    output[idx] = vertex_lookup(hit_positions, hit_primIDs, vbo, ibo, features, idx);
}

[TorchEntryPoint]
TorchTensor<float> mesh_lookup_fwd(
    TorchTensor<float3> hit_positions,  // [n_hits, 3]
    TorchTensor<int> hit_primIDs,       // [n_hits, 1]
    TorchTensor<float3> vbo,            // [n_vertices, 3]
    TorchTensor<int3> ibo,              // [n_faces, 3]
    TorchTensor<float> features         // [n_vertices, n_features]
) {
    var output = TorchTensor<float>.alloc(hit_positions.size(0), features.size(1));
    var d_features = DiffTensorView<float, AtomicAdd>(features);
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((output.size(0) + 1023) / 1024, features.size(1), 1);
    __dispatch_kernel(mesh_lookup_fwd_kernel, blockCount, groupSize)(hit_positions, hit_primIDs, vbo, ibo, d_features, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void mesh_lookup_bwd_kernel(
    TensorView<float3> hit_positions,   // [n_hits, 3]
    TensorView<int> hit_primIDs,        // [n_hits, 1]
    TensorView<float3> vbo,             // [n_vertices, 3]
    TensorView<int3> ibo,               // [n_faces, 3]
    DiffTensorView<float> features,     // [n_vertices, n_features]
    TensorView<float> output_grad
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= hit_positions.size(0) || idx.y >= features.size(1)) return;
    __bwd_diff(vertex_lookup)(hit_positions, hit_primIDs, vbo, ibo, features, idx, output_grad[idx]);
}

[TorchEntryPoint]
TorchTensor<float> mesh_lookup_bwd(
    TorchTensor<float3> hit_positions,  // [n_hits, 3]
    TorchTensor<int> hit_primIDs,       // [n_hits, 1]
    TorchTensor<float3> vbo,            // [n_vertices, 3]
    TorchTensor<int3> ibo,              // [n_faces, 3]
    TorchTensor<float> features,        // [n_vertices, n_features]
    TorchTensor<float> output_grad
) {
    var features_grad = TorchTensor<float>.zerosLike(features);
    var d_features = DiffTensorView<float, AtomicAdd>(features, { features_grad });
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((hit_positions.size(0) + 1023) / 1024, features.size(1), 1);
    __dispatch_kernel(mesh_lookup_bwd_kernel, blockCount, groupSize)(hit_positions, hit_primIDs, vbo, ibo, d_features, output_grad);
    return features_grad;
}

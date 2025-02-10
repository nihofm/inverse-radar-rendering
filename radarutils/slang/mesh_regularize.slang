// -----------------------------------------------
// Mesh sample kernel

[Differentiable]
float sigmoid(const float x) {
    return 1 / (1 + exp(-x));
}

[Differentiable]
float reg_loss(float3 x, float fx, float3 y, float fy) {
    return abs(fx - fy) / max(0.01, dot(y - x, y - x));
}

[Differentiable]
float vertex_regularize(
    no_diff TensorView<float3> vbo, // [n_vertices, 3]
    TensorView<int2> start_end,     // [n_vertices, 2]
    TensorView<int> neighbors,      // [n_vertices * n_neighbors, 1]
    DiffTensorView<float> features, // [n_vertices, n_features]
    uint2 idx
) {
    const float3 x = no_diff vbo[idx.x];
    const float fx = sigmoid(features[idx.x, idx.y]);
    const int2 ij = no_diff start_end[idx.x];
    float loss = 0.f;
    [MaxIters(32)]
    for (int i = ij.x; i < ij.y; ++i) {
        const int neighbor_idx = neighbors[i];
        const float3 y = no_diff vbo[neighbor_idx];
        const float fy = sigmoid(features[neighbor_idx, idx.y]);
        loss += reg_loss(x, fx, y, fy);
    }
    return loss / max(1.f, ij.y - ij.x);
}

// ---------------------------------------------------
// Forward pass

[CudaKernel]
void mesh_regularize_fwd_kernel(
    TensorView<float3> vbo,         // [n_vertices, 3]
    TensorView<int2> start_end,     // [n_vertices, 2]
    TensorView<int> neighbors,      // [n_vertices * n_neighbors, 1]
    DiffTensorView<float> features, // [n_vertices, n_features]
    TensorView<float> output        // [n_vertices, 1]
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= vbo.size(0) || idx.y >= features.size(1)) return;
    output[idx] = vertex_regularize(vbo, start_end, neighbors, features, idx);
}

[TorchEntryPoint]
TorchTensor<float> mesh_regularize_fwd(
    TorchTensor<float3> vbo,     // [n_vertices, 3]
    TorchTensor<int2> start_end, // [n_vertices, 2]
    TorchTensor<int> neighbors,  // [n_vertices * n_neighbors, 1]
    TorchTensor<float> features  // [n_vertices, n_features]
) {
    var output = TorchTensor<float>.alloc(vbo.size(0), features.size(1));
    var d_features = DiffTensorView<float, AtomicAdd>(features);
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((vbo.size(0) + 1023) / 1024, features.size(1), 1);
    __dispatch_kernel(mesh_regularize_fwd_kernel, blockCount, groupSize)(vbo, start_end, neighbors, d_features, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void mesh_regularize_bwd_kernel(
    TensorView<float3> vbo,         // [n_vertices, 3]
    TensorView<int2> start_end,     // [n_vertices, 2]
    TensorView<int> neighbors,      // [n_vertices * n_neighbors, 1]
    DiffTensorView<float> features, // [n_vertices, n_features]
    TensorView<float> output_grad   // [n_vertices, 1]
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= vbo.size(0) || idx.y >= features.size(1)) return;
    __bwd_diff(vertex_regularize)(vbo, start_end, neighbors, features, idx, output_grad[idx]);
}

[TorchEntryPoint]
TorchTensor<float> mesh_regularize_bwd(
    TorchTensor<float3> vbo,       // [n_vertices, 3]
    TorchTensor<int2> start_end,   // [n_vertices, 2]
    TorchTensor<int> neighbors,    // [n_vertices * n_neighbors, 1]
    TorchTensor<float> features,   // [n_vertices, n_features]
    TorchTensor<float> output_grad // [n_vertices, 1]
) {
    var features_grad = TorchTensor<float>.zerosLike(features);
    var d_features = DiffTensorView<float, AtomicAdd>(features, { features_grad });
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((vbo.size(0) + 1023) / 1024, features.size(1), 1);
    __dispatch_kernel(mesh_regularize_bwd_kernel, blockCount, groupSize)(vbo, start_end, neighbors, d_features, output_grad);
    return features_grad;
}

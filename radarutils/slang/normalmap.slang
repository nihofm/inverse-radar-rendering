[Differentiable]
float safe_length(float3 x) {
    return sqrt(max(1e-6f, dot(x, x))); // bwd_diff(sqrt(0)) == NaN
}

[Differentiable]
float3 safe_normalize(float3 x) {
    return x / safe_length(x);
}

[Differentiable] 
float3 compute_tangent(const float3 N) {
    return abs(N.x) > abs(N.y) ?
        float3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z) :
        float3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
}

[Differentiable]
float3 apply_normalmap(DiffTensorView<float> normals, DiffTensorView<float> normalmap, no_diff uint idx) {
    const float3 N = safe_normalize(float3(normals[idx, 0], normals[idx, 1], normals[idx, 2]));
    const float3 N_map = safe_normalize(float3(normalmap[idx, 0], normalmap[idx, 1], normalmap[idx, 2]));
    const float3 T = compute_tangent(N);
    const float3 B = cross(N, T);
    return safe_normalize(N_map.x * T + N_map.y * B + N_map.z * N);
}

// ---------------------------------------------------
// Forward pass

[CudaKernel]
void normalmap_fwd_kernel(
    DiffTensorView<float> normals,
    DiffTensorView<float> normalmap,
    TensorView<float> output
) {
    uint idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (idx >= normals.size(0) || idx >= normalmap.size(0)) return;
    const float3 N = apply_normalmap(normals, normalmap, idx);
    output[idx, 0] = N.x;
    output[idx, 1] = N.y;
    output[idx, 2] = N.z;
}

[TorchEntryPoint]
TorchTensor<float> normalmap_fwd(
    TorchTensor<float> normals,
    TorchTensor<float> normalmap
) {
    var output = TorchTensor<float>.emptyLike(normals);
    var d_normals = DiffTensorView<float, AtomicAdd>(normals);
    var d_normalmap = DiffTensorView<float, AtomicAdd>(normalmap);
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((output.size(0) + 1023) / 1024, 1, 1);
    __dispatch_kernel(normalmap_fwd_kernel, blockCount, groupSize)(d_normals, d_normalmap, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void normalmap_bwd_kernel(
    DiffTensorView<float> normals,
    DiffTensorView<float> normalmap,
    TensorView<float> output_grad
) {
    uint idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (idx >= normals.size(0) || idx >= normalmap.size(0)) return;
    __bwd_diff(apply_normalmap)(normals, normalmap, idx, output_grad[idx]);
}

struct NormalmapGradients {
    TorchTensor<float> normals_grad;
    TorchTensor<float> normalmap_grad;
};

[TorchEntryPoint]
NormalmapGradients normalmap_bwd(
    TorchTensor<float> normals,
    TorchTensor<float> normalmap,
    TorchTensor<float> output_grad
) {
    var normals_grad = TorchTensor<float>.emptyLike(normals);
    var d_normals = DiffTensorView<float, AtomicAdd>(normals, { normals_grad });
    var normalmap_grad = TorchTensor<float>.emptyLike(normals);
    var d_normalmap = DiffTensorView<float, AtomicAdd>(normalmap, { normalmap_grad });
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((normals.size(0) + 1023) / 1024, 1, 1);
    __dispatch_kernel(normalmap_fwd_kernel, blockCount, groupSize)(d_normals, d_normalmap, output_grad);
    return { normals_grad, normalmap_grad };
}

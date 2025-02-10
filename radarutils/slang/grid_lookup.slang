// -----------------------------------------------
// Grid sample kernel

[Differentiable]
float sigmoid(const float x) {
    return 1 / (1 + exp(-x));
}

[Differentiable]
float lookup_grid(DiffTensorView<float> grid, int4 iipos) {
    return sigmoid(grid[clamp(iipos, int4(0), int4(grid.size(0) - 1, grid.size(1) - 1, grid.size(2) - 1, grid.size(3) - 1))]);
}

[Differentiable]
float trilinear_lookup(no_diff TensorView<float3> ipos, DiffTensorView<float> grid, uint2 idx) {
    const float3 ipos_grid = clamp(no_diff ipos[idx.x], 0.f, 1.f) * float3(grid.size(0), grid.size(1), grid.size(2));
    const float3 f = frac(ipos_grid - 0.5);
    const int4 iipos = int4(int3(floor(ipos_grid - 0.5)), idx.y);
    const float lx0 = lerp(lookup_grid(grid, iipos + int4(0, 0, 0, 0)), lookup_grid(grid, iipos + int4(1, 0, 0, 0)), f.x);
    const float hx0 = lerp(lookup_grid(grid, iipos + int4(0, 0, 1, 0)), lookup_grid(grid, iipos + int4(1, 0, 1, 0)), f.x);
    const float lx1 = lerp(lookup_grid(grid, iipos + int4(0, 1, 0, 0)), lookup_grid(grid, iipos + int4(1, 1, 0, 0)), f.x);
    const float hx1 = lerp(lookup_grid(grid, iipos + int4(0, 1, 1, 0)), lookup_grid(grid, iipos + int4(1, 1, 1, 0)), f.x);
    return lerp(lerp(lx0, lx1, f.y), lerp(hx0, hx1, f.y), f.z);
}

// ---------------------------------------------------
// Forward pass

[CudaKernel]
void grid_lookup_fwd_kernel(
    TensorView<float3> ipos,
    DiffTensorView<float> grid,
    TensorView<float> output,
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= ipos.size(0) || idx.y >= grid.size(3)) return;
    output[idx] = trilinear_lookup(ipos, grid, idx);
}

[TorchEntryPoint]
TorchTensor<float> grid_lookup_fwd(
    TorchTensor<float3> ipos,
    TorchTensor<float> grid
) {
    var output = TorchTensor<float>.alloc(ipos.size(0), grid.size(3));
    var d_grid = DiffTensorView<float, AtomicAdd>(grid);
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((ipos.size(0) + 1023) / 1024, grid.size(3), 1);
    __dispatch_kernel(grid_lookup_fwd_kernel, blockCount, groupSize)(ipos, d_grid, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void grid_lookup_bwd_kernel(
    TensorView<float3> ipos,
    DiffTensorView<float> grid,
    TensorView<float> output_grad,
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= ipos.size(0) || idx.y >= grid.size(3)) return;
    __bwd_diff(trilinear_lookup)(ipos, grid, idx, output_grad[idx]);
}

[TorchEntryPoint]
TorchTensor<float> grid_lookup_bwd(
    TorchTensor<float3> ipos,
    TorchTensor<float> grid,
    TorchTensor<float> output_grad,
) {
    var grid_grad = TorchTensor<float>.zerosLike(grid);
    var d_grid = DiffTensorView<float, AtomicAdd>(grid, { grid_grad });
    let groupSize = uint3(1024, 1, 1);
    let blockCount = uint3((ipos.size(0) + 1023) / 1024, grid.size(3), 1);
    __dispatch_kernel(grid_lookup_bwd_kernel, blockCount, groupSize)(ipos, d_grid, output_grad);
    return grid_grad;
}

import helpers;

// -----------------------------------------------
// SFCW reco kernel

// TODO: optimize (tx/rx/freq lookups outside)
[Differentiable]
float2 compute_reco(
    no_diff TensorView<float3> tx_positions,
    no_diff TensorView<float3> rx_positions,
    no_diff TensorView<float> frequencies,
    DiffTensorView<float> signal,
    no_diff const float3 reco_pos,
    no_diff const uint3 idx
) {
    const float3 tx_pos = no_diff tx_positions[idx.x];
    const float3 rx_pos = no_diff rx_positions[idx.y];
    const float delay = (distance(tx_pos, reco_pos) + distance(rx_pos, reco_pos)) / C;
    // correlate actual signal with complex conjugate of expected signal
    const float2 actual = float2(signal[idx.x, idx.y, idx.z, 0], signal[idx.x, idx.y, idx.z, 1]);
    const float phase = 2.f * M_PI * no_diff frequencies[idx.z] * delay;
    float sinp, cosp; sincos(phase, sinp, cosp);
    const float2 expected_conj = float2(cosp, sinp);
    return cmul(actual, expected_conj);
}

// ---------------------------------------------------
// Forward pass

[CudaKernel]
void sfcw_reco_AABB_fwd_kernel(
    TensorView<float3> tx_positions,
    TensorView<float3> rx_positions,
    TensorView<float> frequencies,
    DiffTensorView<float> signal,
    TensorView<float3> AABB,
    TensorView<float2> output
) {
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= output.size(0) || idx.y >= output.size(1) || idx.z >= output.size(2)) return;
    const float3 tc = float3(idx + .5f) / float3(output.size(0), output.size(1), output.size(2));
    const float3 reco_pos = AABB[0] + tc * (AABB[1] - AABB[0]);
    // loop and accumulate over each antenna pair and frequency
    float2 reco = float2(1e-6);
    for (uint tx_idx = 0; tx_idx < tx_positions.size(0); ++tx_idx)
        for (uint rx_idx = 0; rx_idx < rx_positions.size(0); ++rx_idx)
            for (uint t_idx = 0; t_idx < frequencies.size(0); ++t_idx)
                reco += compute_reco(tx_positions, rx_positions, frequencies, signal, reco_pos, uint3(tx_idx, rx_idx, t_idx));
    output[idx.x, idx.y, idx.z] = reco;
}

[TorchEntryPoint]
TorchTensor<float2> sfcw_reco_AABB_fwd(
    TorchTensor<float3> tx_positions,
    TorchTensor<float3> rx_positions,
    TorchTensor<float> frequencies,
    TorchTensor<float> signal,
    TorchTensor<float3> AABB,
    uint n_voxels_xy,
    uint n_voxels_z
) {
    var output = TorchTensor<float2>.alloc(n_voxels_xy, n_voxels_xy, n_voxels_z);
    var d_signal = DiffTensorView<float, AtomicAdd>(signal);
    let threads = uint3(8, 8, 8);
    let blocks = uint3((output.size(0) + 7) / 8, (output.size(1) + 7) / 8, (output.size(2) + 7) / 8);
    __dispatch_kernel(sfcw_reco_AABB_fwd_kernel, blocks, threads)(tx_positions, rx_positions, frequencies, d_signal, AABB, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void sfcw_reco_AABB_bwd_kernel(
    TensorView<float3> tx_positions,
    TensorView<float3> rx_positions,
    TensorView<float> frequencies,
    DiffTensorView<float> signal,
    TensorView<float3> AABB,
    TensorView<float2> output_grad
) {
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= output_grad.size(0) || idx.y >= output_grad.size(1) || idx.z >= output_grad.size(2)) return;
    const float3 tc = float3(idx + .5f) / float3(output_grad.size(0), output_grad.size(1), output_grad.size(2));
    const float3 reco_pos = AABB[0] + tc * (AABB[1] - AABB[0]);
    // loop and accumulate over each antenna pair and frequency
    float2 reco = float2(1e-6);
    for (uint tx_idx = 0; tx_idx < tx_positions.size(0); ++tx_idx)
        for (uint rx_idx = 0; rx_idx < rx_positions.size(0); ++rx_idx)
            for (uint t_idx = 0; t_idx < frequencies.size(0); ++t_idx)
                bwd_diff(compute_reco)(tx_positions, rx_positions, frequencies, signal, reco_pos, uint3(tx_idx, rx_idx, t_idx), output_grad[idx]);
}

[TorchEntryPoint]
TorchTensor<float> sfcw_reco_AABB_bwd(
    TorchTensor<float3> tx_positions,
    TorchTensor<float3> rx_positions,
    TorchTensor<float> frequencies,
    TorchTensor<float> signal,
    TorchTensor<float3> AABB,
    TorchTensor<float2> output_grad,
) {
    var signal_grad = TorchTensor<float>.zerosLike(signal);
    var d_signal = DiffTensorView<float, AtomicAdd>(signal, { signal_grad });
    let threads = uint3(8, 8, 8);
    let blocks = uint3((output_grad.size(0) + 7) / 8, (output_grad.size(1) + 7) / 8, (output_grad.size(2) + 7) / 8);
    __dispatch_kernel(sfcw_reco_AABB_bwd_kernel, blocks, threads)(tx_positions, rx_positions, frequencies, d_signal, AABB, output_grad);
    return signal_grad;
}

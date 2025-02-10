import helpers;

static const uint kMaterialType = MATERIAL_TYPE;        // compile-time switch between material types
static const bool kWithAntennaPattern = bool(USE_APC);  // compile-time switch to enable APC
static const float kMinFrequency = 72e9;                // hardcoded
static const float kMaxFrequency = 82e9;                // hardcoded
static const float kBandwidth = kMaxFrequency - kMinFrequency;

// ---------------------------------------------------
// Antenna pattern eval helper

float antenna_pattern_lookup_gain(const float3 w, TensorView<float> pattern) {
    const float2 thetha_phi = to_spherical(w);
    const float2 ipos = float2(pattern.size(0) * thetha_phi.y / M_PI, pattern.size(1) * thetha_phi.x / (2 * M_PI));
    const float2 f = frac(ipos - 0.5);
    const int3 iipos = int3(int2(ipos - 0.5), 0);
    const int3 size = int3(pattern.size(0), pattern.size(1), pattern.size(2));
    const float y0 = lerp(pattern[(iipos + int3(0, 0, 0)) % size], pattern[(iipos + int3(1, 0, 0)) % size], f.x);
    const float y1 = lerp(pattern[(iipos + int3(0, 1, 0)) % size], pattern[(iipos + int3(1, 1, 0)) % size], f.x);
    return lerp(y0, y1, f.y);
}

// ---------------------------------------------------
// Simple reflectivity material

[Differentiable]
float2 eval_material_baseline(
    const float3 tx_pos,
    const float3 rx_pos,
    const float frequency,
    TensorView<float> tx_pattern,
    TensorView<float> rx_pattern,
    const float3 P,
    const float3 N,
    const float3 w_o,
    const float3 w_i,
    const float4 params,
) {
    // "material"
    const float f_material = params.x;
    // compute path loss
    const float d_tx = distance(tx_pos, P);
    const float d_rx = distance(rx_pos, P);
    const float throughput = f_material * max(0.f, dot(N, w_i)) * max(0.f, dot(N, w_o)) / max(EPS, pow(d_tx, 2) * pow(d_rx, 2));
    // compute phase
    const float phase = -2 * M_PI * (d_tx + d_rx) * frequency / C;
    float sinp, cosp; sincos(phase, sinp, cosp);
    // compute signal
    float gain = 1.f;
    if (kWithAntennaPattern)
        gain = no_diff antenna_pattern_lookup_gain(-w_i, tx_pattern) * no_diff antenna_pattern_lookup_gain(-w_o, rx_pattern);
    return gain * throughput * float2(cosp, sinp);
}

// ---------------------------------------------------
// diffuse + specular (blinn-phong) material, adapted from Schuessler et al. for NEE

[Differentiable]
float2 eval_material_phong(
    const float3 tx_pos,
    const float3 rx_pos,
    const float frequency,
    TensorView<float> tx_pattern,
    TensorView<float> rx_pattern,
    const float3 P,
    const float3 N,
    const float3 w_o,
    const float3 w_i,
    const float4 params,
) {
    const float cos_wo = dot(N, w_o);
    const float cos_wi = dot(N, w_i);
    const float3 H = safe_normalize(w_o + w_i);
    const float cos_h = dot(N, H);
    if (cos_wo <= 0.f || cos_wi <= 0.f || cos_h <= 0.f) return 0;
    const float f_alpha = params.x;
    const float f_diffuse = INVPI;
    const float exponent = 1 + params.y * 100;
    const float f_specular = pow(cos_h, exponent) * (exponent + 1) * INV2PI;
    const float f_material = lerp(f_diffuse, f_specular, f_alpha); // mix diffuse and specular (blinn-phong) lobes
    // compute path loss
    const float d_tx = distance(tx_pos, P);
    const float d_rx = distance(rx_pos, P);
    const float throughput = f_material * max(0.f, dot(N, w_i)) * max(0.f, dot(N, w_o)) / max(EPS, pow(d_tx, 2) * pow(d_rx, 2));
    // compute phase
    const float phase = -2 * M_PI * (d_tx + d_rx) * frequency / C;
    float sinp, cosp; sincos(phase, sinp, cosp);
    // compute signal
    float gain = 1.f;
    if (kWithAntennaPattern)
        gain = no_diff antenna_pattern_lookup_gain(-w_i, tx_pattern) * no_diff antenna_pattern_lookup_gain(-w_o, rx_pattern);
    return gain * throughput * float2(cosp, sinp);
}

// ---------------------------------------------------
// specular GGX material

[Differentiable]
float fresnel_dielectric(const float cos_i, const float n1, const float n2) {
    const float sin_t = (n1 / n2) * sqrt(max(EPS, 1.f - cos_i * cos_i));
    const float cos_t = sqrt(max(EPS, 1.f - sin_t * sin_t));
    return sqr((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t));
}

[Differentiable]
float GGX_D(const float cos_h, float roughness) {
    const float tan2 = tan2_t(cos_h);
    const float a2 = sqr(roughness);
    return a2 / (M_PI * sqr(sqr(cos_h)) * sqr(a2 + tan2));
}

[Differentiable]
float GGX_G1(const float cos_w, float roughness) {
    const float tan2 = tan2_t(cos_w);
    return 2.f / (1.f + sqrt(1.f + sqr(roughness) * tan2));
}

[Differentiable]
float2 eval_material_layered_ggx(
    const float3 tx_pos,
    const float3 rx_pos,
    const float frequency,
    TensorView<float> tx_pattern,
    TensorView<float> rx_pattern,
    const float3 P,
    const float3 N,
    const float3 w_o,
    const float3 w_i,
    const float4 params,
) {
    const float cos_wo = dot(N, w_o);
    const float cos_wi = dot(N, w_i);
    const float3 H = safe_normalize(w_o + w_i);
    const float cos_h = dot(N, H);
    if (cos_wo <= 0.f || cos_wi <= 0.f || cos_h <= 0.f) return 0;
    const float F = fresnel_dielectric(dot(H, w_i), 1.0, 1.0 + params.y * 10);
    const float roughness = max(0.01f, params.z); // avoid singularity at 0
    const float D = GGX_D(cos_h, roughness);
    const float G = GGX_G1(cos_wo, roughness) * GGX_G1(cos_wi, roughness);
    const float f_ggx = (F * D * G) / (4 * cos_wi * cos_wo);
    const float f_material = lerp(params.x / M_PI, f_ggx, params.w); // mix (absorbing) diffuse and ggx lobes
    // compute path loss
    const float d_tx = distance(tx_pos, P);
    const float d_rx = distance(rx_pos, P);
    const float throughput = f_material * max(0.f, dot(N, w_i)) * max(0.f, dot(N, w_o)) / max(EPS, pow(d_tx, 2) * pow(d_rx, 2));
    // compute phase
    const float phase = -2 * M_PI * (d_tx + d_rx) * frequency / C;
    float sinp, cosp; sincos(phase, sinp, cosp);
    // compute signal
    float gain = 1.f;
    if (kWithAntennaPattern)
        gain = no_diff antenna_pattern_lookup_gain(-w_i, tx_pattern) * no_diff antenna_pattern_lookup_gain(-w_o, rx_pattern);
    return gain * throughput * float2(cosp, sinp);
}

// ---------------------------------------------------
// Smooth fresnel material

[Differentiable]
float2 eval_material_fresnel_smooth(
    const float3 tx_pos,
    const float3 rx_pos,
    const float frequency,
    TensorView<float> tx_pattern,
    TensorView<float> rx_pattern,
    const float3 P,
    const float3 N,
    const float3 w_o,
    const float3 w_i,
    const float4 params,
) {
    const float cos_wi = dot(N, w_i);
    if (cos_wi <= 0.f) return 0.f;
    // evaluate fresnel term
    const float n = 1 + params.x * 99;
    const float k = 0 + params.y * 100;
    const float4 Fr_sp = fresnel_r_sp_amplitude(cos_wi, float2(n, -k));
    // compute path loss
    const float d_tx = distance(tx_pos, P);
    const float d_rx = distance(rx_pos, P);
    const float throughput = max(0.f, dot(N, w_i)) * max(0.f, dot(N, w_o)) / max(EPS, pow(d_tx, 2) * pow(d_rx, 2));
    // compute phase
    const float phase = -2 * M_PI * (d_tx + d_rx) * frequency / C;
    float sinp, cosp; sincos(phase, sinp, cosp);
    // compute signal
    float gain = 1.f;
    if (kWithAntennaPattern)
        gain = no_diff antenna_pattern_lookup_gain(-w_i, tx_pattern) * no_diff antenna_pattern_lookup_gain(-w_o, rx_pattern);
    return gain * throughput * lerp(cmul(Fr_sp.xy, float2(cosp, sinp)), cmul(Fr_sp.zw, float2(cosp, sinp)), params.w);
}

// ---------------------------------------------------
// Rough fresnel material w/o conductivity

[Differentiable]
float2 eval_material_fresnel_rough(
    const float3 tx_pos,
    const float3 rx_pos,
    const float frequency,
    TensorView<float> tx_pattern,
    TensorView<float> rx_pattern,
    const float3 P,
    const float3 N,
    const float3 w_o,
    const float3 w_i,
    const float4 params,
) {
    const float cos_wi = dot(N, w_i);
    const float cos_wo = dot(N, w_o);
    const float3 H = safe_normalize(w_o + w_i);
    const float cos_h = dot(N, H);
    if (cos_wo <= 0.f || cos_wi <= 0.f || cos_h <= 0.f) return 0.f;
    // evaluate fresnel term
    const float n = 1 + params.x * 99;
    const float k = 0 + params.y * 100;
    const float4 Fr_sp = fresnel_r_sp_amplitude(cos_wi, float2(n, -k));
    // evaluate GGX roughness term
    const float roughness = max(0.01f, params.z); // avoid singularity at 0
    const float D = GGX_D(cos_h, roughness);
    const float G = GGX_G1(cos_wo, roughness) * GGX_G1(cos_wi, roughness);
    const float f_material = (D * G) / (4 * cos_wi * cos_wo); // note the missing F term -> Fr_sp
    // compute path loss
    const float d_tx = distance(tx_pos, P);
    const float d_rx = distance(rx_pos, P);
    const float throughput = f_material * max(0.f, dot(N, w_i)) * max(0.f, dot(N, w_o)) / max(EPS, pow(d_tx, 2) * pow(d_rx, 2));
    // compute phase
    const float phase = -2 * M_PI * (d_tx + d_rx) * frequency / C;
    float sinp, cosp; sincos(phase, sinp, cosp);
    // compute signal
    float gain = 1.f;
    if (kWithAntennaPattern)
        gain = no_diff antenna_pattern_lookup_gain(-w_i, tx_pattern) * no_diff antenna_pattern_lookup_gain(-w_o, rx_pattern);
    return gain * throughput * lerp(cmul(Fr_sp.xy, float2(cosp, sinp)), cmul(Fr_sp.zw, float2(cosp, sinp)), params.w);
}

// ---------------------------------------------------
// Main signal computation function

[Differentiable]
float2 compute_signal_material(
    no_diff float3 tx_pos,
    no_diff float3 rx_pos,
    no_diff float frequency,
    no_diff TensorView<float> tx_pattern,
    no_diff TensorView<float> rx_pattern,
    float3 P,
    float3 N,
    float4 params0,
    float4 params1
) {
    // interpolate material params and compute w_o and w_i
    const float4 params = lerp(params0, params1, (frequency - kMinFrequency) / kBandwidth);
    const float3 w_o = safe_normalize(rx_pos - P);
    const float3 w_i = safe_normalize(tx_pos - P);
    // call material function using compile-time constant
    switch (kMaterialType) {
    default:
    case 0:
        return eval_material_baseline(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, P, N, w_o, w_i, params);
    case 1:
        return eval_material_phong(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, P, N, w_o, w_i, params);
    case 2:
        return eval_material_layered_ggx(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, P, N, w_o, w_i, params);
    case 3:
        return eval_material_fresnel_smooth(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, P, N, w_o, w_i, params);
    case 4:
        return eval_material_fresnel_rough(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, P, N, w_o, w_i, params);
    }
}

// ---------------------------------------------------
// Forward pass

[CudaKernel]
void sfcw_signal_material_fwd_kernel(
    TensorView<float3> tx_positions,  // [n_tx, 3]
    TensorView<float3> rx_positions,  // [n_rx, 3]
    TensorView<float> frequencies,    // [n_freq, 1]
    TensorView<float> tx_pattern,     // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TensorView<float> rx_pattern,     // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TensorView<float3> hit_positions, // [n_hits, 3]
    TensorView<float3> hit_normals,   // [n_hits, 3]
    TensorView<float> hit_material,   // [n_hits, n_features]
    TensorView<float2> output,        // [n_tx, n_rx, n_freq, 2]
) {
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= tx_positions.size(0) || idx.y >= rx_positions.size(0) || idx.z >= frequencies.size(0)) return;
    float3 tx_pos = tx_positions[idx.x];
    float3 rx_pos = rx_positions[idx.y];
    float frequency = frequencies[idx.z];
    float2 signal = float2(1e-6); // complex number
    for (int hit_idx = 0; hit_idx < hit_positions.size(0); ++hit_idx) {
        const float3 P = hit_positions[hit_idx];
        const float3 N = hit_normals[hit_idx];
        const float4 params0 = float4(hit_material[hit_idx, 0], hit_material[hit_idx, 2], hit_material[hit_idx, 4], hit_material[hit_idx, 6]);
        const float4 params1 = float4(hit_material[hit_idx, 1], hit_material[hit_idx, 3], hit_material[hit_idx, 5], hit_material[hit_idx, 7]);
        signal += compute_signal_material(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, P, N, params0, params1);
    }
    output[idx] = signal;
}

[TorchEntryPoint]
TorchTensor<float2> sfcw_signal_material_fwd(
    TorchTensor<float3> tx_positions,  // [n_tx, 3]
    TorchTensor<float3> rx_positions,  // [n_rx, 3]
    TorchTensor<float> frequencies,    // [n_freq, 1]
    TorchTensor<float> tx_pattern,     // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TorchTensor<float> rx_pattern,     // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TorchTensor<float3> hit_positions, // [n_hits, 3]
    TorchTensor<float3> hit_normals,   // [n_hits, 3]
    TorchTensor<float> hit_material,   // [n_hits, n_features]
) {
    var output = TorchTensor<float2>.alloc(tx_positions.size(0), rx_positions.size(0), frequencies.size(0));
    let threads = uint3(8, 8, 8);
    let blocks = uint3((output.size(0) + 7) / 8, (output.size(1) + 7) / 8, (output.size(2) + 7) / 8);
    __dispatch_kernel(sfcw_signal_material_fwd_kernel, blocks, threads)(tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_normals, hit_material, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void sfcw_signal_material_bwd_kernel(
    TensorView<float3> tx_positions,       // [n_tx, 3]
    TensorView<float3> rx_positions,       // [n_rx, 3]
    TensorView<float> frequencies,         // [n_freq, 1]
    TensorView<float> tx_pattern,          // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TensorView<float> rx_pattern,          // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TensorView<float3> hit_positions,      // [n_hits, 3]
    TensorView<float3> hit_positions_grad, // [n_hits, 3]
    TensorView<float3> hit_normals,        // [n_hits, 3]
    TensorView<float3> hit_normals_grad,   // [n_hits, 3]
    TensorView<float> hit_material,        // [n_hits, n_features]
    TensorView<float> hit_material_grad,   // [n_hits, n_features]
    TensorView<float2> output_grad,        // [n_tx, n_rx, n_freq, 2]
) {
    uint idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (idx >= hit_positions.size(0)) return;
    // perform lookups and init gradients
    const float3 P = hit_positions[idx];
    const float3 N = hit_normals[idx];
    const float4 params0 = float4(hit_material[idx, 0], hit_material[idx, 2], hit_material[idx, 4], hit_material[idx, 6]);
    const float4 params1 = float4(hit_material[idx, 1], hit_material[idx, 3], hit_material[idx, 5], hit_material[idx, 7]);
    float3 P_grad = float3(0);
    float3 N_grad = float3(0);
    float4 params0_grad = float4(0);
    float4 params1_grad = float4(0);
    // compute and accumulate gradients (transposed from fwd to avoid atomics)
    for (int tx_idx = 0; tx_idx < tx_positions.size(0); ++tx_idx) {
        float3 tx_pos = tx_positions[tx_idx];
        for (int rx_idx = 0; rx_idx < rx_positions.size(0); ++rx_idx) {
            float3 rx_pos = rx_positions[rx_idx];
            for (int f_idx = 0; f_idx < frequencies.size(0); ++f_idx) {
                float frequency = frequencies[f_idx];
                DifferentialPair<float3> d_P = DifferentialPair<float3>(P, float3(0));
                DifferentialPair<float3> d_N = DifferentialPair<float3>(N, float3(0));
                DifferentialPair<float4> d_params0 = DifferentialPair<float4>(params0, float4(0));
                DifferentialPair<float4> d_params1 = DifferentialPair<float4>(params1, float4(0));
                __bwd_diff(compute_signal_material)(tx_pos, rx_pos, frequency, tx_pattern, rx_pattern, d_P, d_N, d_params0, d_params1, output_grad[uint3(tx_idx, rx_idx, f_idx)]);
                P_grad += d_P.d;
                N_grad += d_N.d;
                params0_grad += d_params0.d;
                params1_grad += d_params1.d;
            }
        }
    }
    // store accumulated gradients
    hit_positions_grad[idx] = P_grad;
    hit_normals_grad[idx] = N_grad;
    hit_material_grad[idx, 0] = params0_grad.x;
    hit_material_grad[idx, 1] = params1_grad.x;
    hit_material_grad[idx, 2] = params0_grad.y;
    hit_material_grad[idx, 3] = params1_grad.y;
    hit_material_grad[idx, 4] = params0_grad.z;
    hit_material_grad[idx, 5] = params1_grad.z;
    hit_material_grad[idx, 6] = params0_grad.w;
    hit_material_grad[idx, 7] = params1_grad.w;
}

struct SignalMaterialGrads {
    TorchTensor<float3> hit_positions_grad;
    TorchTensor<float3> hit_normals_grad;
    TorchTensor<float> hit_material_grad;
};

[TorchEntryPoint]
SignalMaterialGrads sfcw_signal_material_bwd(
    TorchTensor<float3> tx_positions,  // [n_tx, 3]
    TorchTensor<float3> rx_positions,  // [n_rx, 3]
    TorchTensor<float> frequencies,    // [n_freq, 1]
    TorchTensor<float> tx_pattern,     // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TorchTensor<float> rx_pattern,     // [n_phi, n_theta, 5] -> (G_abs, G_s, phase_s, G_p, phase_p)
    TorchTensor<float3> hit_positions, // [n_hits, 3]
    TorchTensor<float3> hit_normals,   // [n_hits, 3]
    TorchTensor<float> hit_material,   // [n_hits, n_features]
    TorchTensor<float2> output_grad,   // [n_tx, n_rx, n_freq, 2]
) {
    var hit_positions_grad = TorchTensor<float3>.zerosLike(hit_positions);
    var hit_normals_grad = TorchTensor<float3>.zerosLike(hit_normals);
    var hit_material_grad = TorchTensor<float>.zerosLike(hit_material);
    let threads = uint3(256, 1, 1);
    let blocks = uint3((hit_positions.size(0) + threads.x - 1) / threads.x, 1, 1);
    __dispatch_kernel(sfcw_signal_material_bwd_kernel, blocks, threads)(tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_positions_grad, hit_normals, hit_normals_grad, hit_material, hit_material_grad, output_grad);
    return { hit_positions_grad, hit_normals_grad, hit_material_grad };
}

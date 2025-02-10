// ---------------------------------------------------
// Constants

static const float M_PI = 3.14159265358979323846;   // pi
static const float INVPI = 1.0 / M_PI;              // 1 / (2*pi)
static const float INV2PI = 1.0 / (2 * M_PI);       // 1 / (2*pi)
static const float INV4PI = 1.0 / (4 * M_PI);       // 1 / (4*pi)
static const float EPS = 1e-6f;                     // float epsilon
static const float C = 299792458;                   // speed of light [m/s]
static const float E0 = 8.8541878128e-12;           // permittivity of free space [F/m]
static const float M0 = 1.25663706212e-6;           // permeability of free space [N/A^2]
static const float Z0 = sqrt(M0 / E0);              // wave impedance of free space [Ohm]

// ---------------------------------------------------
// Trigonometry helpers

[Differentiable] float sqr(float x) { return x * x; }

[Differentiable] float sin_t(float cos_t) { return sqrt(sin2_t(cos_t)); }

[Differentiable] float sin2_t(float cos_t) { return max(0.f, 1.f - sqr(cos_t)); }

[Differentiable] float tan_t(float cos_t) { return sin_t(cos_t) / cos_t; }

[Differentiable] float tan2_t(float cos_t) { return sin2_t(cos_t) / sqr(cos_t); }

// -----------------------------------------------
// Vector helpers

[Differentiable] float safe_length(float3 x) { return sqrt(max(EPS, dot(x, x))); } // bwd_diff(sqrt(0)) == NaN

[Differentiable] float3 safe_normalize(float3 x) { return x / safe_length(x); }

// -----------------------------------------------
// Spherical coordinates helpers

[Differentiable] float3 to_cartesian(const float2 theta_phi) {
    return float3(sin(theta_phi.x) * cos(theta_phi.y), sin(theta_phi.x) * sin(theta_phi.y), cos(theta_phi.x));
}

[Differentiable] float2 to_spherical(const float3 xyz) {
    const float3 dir = safe_normalize(xyz);
    const float theta = acos(clamp(dir.z, -1, 1));
    const float phi = atan2(dir.y, dir.x);
    return float2(theta, phi < 0 ? phi + 2 * M_PI : phi);
}

// -----------------------------------------------
// Complex number helpers

// complex/complex add/sub/mul/div operands
[Differentiable] float2 cadd(const float2 ca, const float2 cb) {
     return ca + cb;
}
[Differentiable] float2 csub(const float2 ca, const float2 cb) {
    return ca - cb;
}
[Differentiable] float2 cmul(const float2 ca, const float2 cb) {
    return float2(ca.x * cb.x - ca.y * cb.y, ca.x * cb.y + ca.y * cb.x);
}
[Differentiable] float2 cdiv(const float2 ca, const float2 cb) {
    const float denom = max(0.001f, cb.x * cb.x + cb.y * cb.y);
    return float2((ca.x * cb.x + ca.y * cb.y) / denom, (ca.y * cb.x - ca.x * cb.y) / denom);
}

// complex operands
[Differentiable] float cabs(const float2 c) {
    return sqrt(c.x * c.x + c.y * c.y);
}
[Differentiable] float2 csqr(const float2 c) {
    return float2(c.x * c.x - c.y * c.y, 2 * c.x * c.y);
}
[Differentiable] float2 csqrt(const float2 c) {
    const float r = sqrt(c.x * c.x + c.y * c.y);
    return sqrt(r) * cadd(c, r) / max(0.001f, cabs(cadd(c, r)));
}
[Differentiable] float2 crcp(const float2 c) {
    const float denom = max(0.001f, c.x * c.x + c.y * c.y);
    return float2(c.x / denom, -c.y / denom);
}
[Differentiable] float2 cconj(const float2 c) {
    return float2(c.x, -c.y);
}

// real min/max
[Differentiable] float2 cmin(const float r, const float2 c) {
    return float2(min(r, c.x), c.y);
}
[Differentiable] float2 cmax(const float r, const float2 c) {
    return float2(max(r, c.x), c.y);
}

// complex/real convenience operands
[Differentiable] float2 cadd(const float2 c, const float r) { return cadd(c, float2(r, 0)); }
[Differentiable] float2 cadd(const float r, const float2 c) { return cadd(float2(r, 0), c); }
[Differentiable] float2 csub(const float2 c, const float r) { return csub(c, float2(r, 0)); }
[Differentiable] float2 csub(const float r, const float2 c) { return csub(float2(r, 0), c); }
[Differentiable] float2 cmul(const float2 c, const float r) { return cmul(c, float2(r, 0)); }
[Differentiable] float2 cmul(const float r, const float2 c) { return cmul(float2(r, 0), c); }
[Differentiable] float2 cdiv(const float2 c, const float r) { return cdiv(c, float2(r, 0)); }
[Differentiable] float2 cdiv(const float r, const float2 c) { return cdiv(float2(r, 0), c); }

// ---------------------------------------------------
// Fresnel helpers

[Differentiable]
float4 fresnel_r_sp_amplitude(float cos_t, float2 m, float2 n_i = float2(1.000273, 0)) {
    float2 m2 = csqr(cdiv(m, n_i));
    float c = cos_t;
    float s = sin_t(cos_t);
    float2 d = csqrt(csub(m2, s * s));
    if (m.y == 0.f) d = cconj(d);
    float2 r_s = cdiv(csub(c, d), cadd(c, d));
    float2 r_p = cdiv(csub(cmul(m2, c), d), cadd(cmul(m2, c), d));
    return float4(r_s, r_p);
}

[Differentiable]
float4 fresnel_t_sp_amplitude(float cos_t, float2 m, float2 n_i = float2(1.000273, 0)) {
    float2 m2 = csqr(cdiv(m, n_i));
    float c = cos_t;
    float s = sin_t(cos_t);
    float2 d = csqrt(csub(m2, s * s));
    if (m.y == 0.f) d = cconj(d);
    float2 t_s = cdiv(cdiv(cmul(2, d), cdiv(m, n_i)), cadd(c, d));
    float2 t_p = cmul(2 * c, cdiv(cdiv(m, n_i), cadd(cmul(m2, c), d)));
    return float4(t_s, t_p);
}

[Differentiable]
float2 fresnel_r_sp_power(float cos_t, float2 m) {
    float4 r_sp = fresnel_r_sp_amplitude(cos_t, m);
    return float2(sqr(cabs(r_sp.xy)), sqr(cabs(r_sp.zw)));
}

[Differentiable]
float2 fresnel_t_sp_power(float cos_t, float2 m, float2 n_i = float2(1.000273, 0)) {
    float2 m2 = csqr(cdiv(m, n_i));
    float c = cos_t;
    float s = sin_t(cos_t);
    float2 d = csqrt(csub(m2, s * s));
    if (m.y == 0.f) d = cconj(d);
    float2 t_s = cdiv(2 * c, cadd(c, d));
    float T_s = cabs(cmul(cdiv(d, c), sqr(cabs(t_s))));
    float2 t_p = cdiv(cmul(2 * c, cdiv(m, n_i)), cadd(cmul(m2, c), d));
    float T_p = cabs(cmul(cdiv(d, c), sqr(cabs(t_p))));
    return float2(T_s, T_p);
}

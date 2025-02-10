#include "context.h"
#include "common.h"
#include "params.h"

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

#include <optix/optix_stubs.h>
#include <optix/optix_stack_size.h>
#include <optix/optix_function_table_definition.h>

#include <ATen/cuda/CUDAUtils.h>

// -----------------------------------------------------------------------------
// helper functions

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

static std::string read_file(const std::filesystem::path& filename) {
    std::ifstream file(filename, std::ios::binary);
    if(file.good()) {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        std::string result;
        result.assign(buffer.begin(), buffer.end());
        return result;
    }
    throw std::runtime_error("Unable to open source file: " + filename.string());
}

static std::string compile_ptx(const std::string& name, const std::string& source_cu, const std::vector<std::string>& includes) {
    // create nvrtc program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_THROW(nvrtcCreateProgram(&prog, source_cu.c_str(), name.c_str(), 0, NULL, NULL));

    // gather NVRTC options
    std::vector<std::string> options;
    for (const auto& path : includes)
        options.push_back("-I " + path);
    options.push_back("-std=c++11");
    options.push_back("-use_fast_math");
    options.push_back("-lineinfo");
    // TODO: more options?

    // compile PTX
    std::vector<const char*> nvrtc_opts;
    for (const auto& opt : options)
        nvrtc_opts.push_back(opt.c_str());
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)nvrtc_opts.size(), nvrtc_opts.data());

    // fetch log output
    std::string nvrtc_log;
    size_t log_size = 0;
    NVRTC_CHECK_THROW(nvrtcGetProgramLogSize(prog, &log_size));
    nvrtc_log.resize(log_size);
    if (log_size > 1)
        NVRTC_CHECK_THROW(nvrtcGetProgramLog(prog, &nvrtc_log[0]));
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error(std::string(FILE_LINE) + ": NVRTC compilation failed.\n" + nvrtc_log);

    // get ptx from compiled program
    std::string ptx;
    size_t ptx_size = 0;
    NVRTC_CHECK_THROW(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_THROW(nvrtcGetPTX(prog, &ptx[0]));

    // destroy nvrtc program
    NVRTC_CHECK_THROW(nvrtcDestroyProgram(&prog));

    return ptx;
}

struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void create_pipeline(OptixDeviceContext context, const char* ptx_data, size_t size, size_t n_payloads, OptixPipeline& pipeline, OptixShaderBindingTable& sbt) {
    char log[2048];
    size_t sizeof_log = sizeof(log);

    // create module from PTX
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur         = false;
    pipeline_compile_options.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline_compile_options.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.numPayloadValues       = n_payloads;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OPTIX_CHECK_THROW_LOG(
        optixModuleCreate(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx_data,
            size,
            log,
            &sizeof_log,
            &module));

    // ceate program groups
    OptixProgramGroup raygen_prog_group             = nullptr;
    OptixProgramGroup miss_prog_group               = nullptr;
    OptixProgramGroup hitgroup_prog_group           = nullptr;
    OptixProgramGroupOptions program_group_options  = {}; // Initialize to zeros
    OptixProgramGroupDesc raygen_prog_group_desc    = {}; // Initialize to zeros

    // raygen program group
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
        context,
        &raygen_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &raygen_prog_group
    ));

    // miss program group
    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
        context,
        &miss_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &miss_prog_group
    ));

    // hit program group
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
        context,
        &hitgroup_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &hitgroup_prog_group
    ));

    // link pipeline
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;

    OPTIX_CHECK_THROW_LOG(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(OptixProgramGroup),
        log,
        &sizeof_log,
        &pipeline
    ));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
        OPTIX_CHECK_THROW(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK_THROW(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        0,  // maxCCDepth
        0,  // maxDCDEpth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));
    OPTIX_CHECK_THROW(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        1  // maxTraversableDepth
    ));

    // prepare shader binding table
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(SbtRecord);
    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
    SbtRecord rg_sbt;
    OPTIX_CHECK_THROW(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK_THROW(cudaMemcpy(
        reinterpret_cast<void*>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(SbtRecord);
    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
    SbtRecord ms_sbt;
    OPTIX_CHECK_THROW(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK_THROW(cudaMemcpy(
        reinterpret_cast<void*>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(SbtRecord);
    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    SbtRecord hg_sbt;
    OPTIX_CHECK_THROW(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK_THROW(cudaMemcpy(
        reinterpret_cast<void*>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));

    sbt = {};
    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof(SbtRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt.hitgroupRecordCount         = 1;

    return;
}

// -----------------------------------------------------------------------------
// OptixContext

OptixContext::OptixContext() : context(0), raytrace_pipeline(0), visibility_pipeline(0), d_gas_output_buffer(0) {
    // init cuda and optix
    CUDA_CHECK_THROW(cudaFree(nullptr));
    OPTIX_CHECK_THROW(optixInit());

    // set context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 3;
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;

    // Associate CUDA and OptiX context
    CUcontext cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK_THROW(optixDeviceContextCreate(cu_ctx, &options, &context));

    // collect include paths
    const std::filesystem::path base_path = STRINGIZE(EXTENSION_BASE_DIRECTORY);
    std::vector<std::string> includes;
    // Windows requires explicit conversion to string
    includes.push_back(base_path.string());
    includes.push_back((base_path / "c_src").string());
    includes.push_back((base_path / "include").string());
    includes.push_back((base_path / "include" / "optix").string());

    // create raytrace pipeline
    const std::string raytrace_source = read_file(base_path / "c_src" / "raytrace.cu");
    const std::string raytrace_ptx = compile_ptx("raytrace", raytrace_source, includes);
    create_pipeline(context, raytrace_ptx.data(), raytrace_ptx.size(), 5, raytrace_pipeline, raytrace_sbt);

    // create visbility pipeline
    const std::string visibility_source = read_file(base_path / "c_src" / "visibility.cu");
    const std::string visibility_ptx = compile_ptx("visibility", visibility_source, includes);
    create_pipeline(context, visibility_ptx.data(), visibility_ptx.size(), 1, visibility_pipeline, visibility_sbt);

    // create hitgen pipeline
    const std::string hitgen_source = read_file(base_path / "c_src" / "hitgen.cu");
    const std::string hitgen_ptx = compile_ptx("hitgen", hitgen_source, includes);
    create_pipeline(context, hitgen_ptx.data(), hitgen_ptx.size(), 5, hitgen_pipeline, hitgen_sbt);
}

OptixContext::~OptixContext() {
    CUDA_CHECK_PRINT(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
    OPTIX_CHECK_PRINT(optixPipelineDestroy(raytrace_pipeline));
    OPTIX_CHECK_PRINT(optixPipelineDestroy(visibility_pipeline));
    OPTIX_CHECK_PRINT(optixPipelineDestroy(hitgen_pipeline));
    OPTIX_CHECK_PRINT(optixDeviceContextDestroy(context));
}

void OptixContext::build_bvh(at::Tensor vertex_buffer) {
    build_bvh(vertex_buffer, at::Tensor());
}

void OptixContext::build_bvh(at::Tensor vertex_buffer, at::Tensor index_buffer) {
    CHECK_TENSOR(vertex_buffer, 2, 3, at::kFloat);
    if (index_buffer.numel()) CHECK_TENSOR(index_buffer, 2, 3, at::kInt);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // free old data on rebuild
    if (d_gas_output_buffer)
        CUDA_CHECK_THROW(cudaFreeAsync(reinterpret_cast<void*>(d_gas_output_buffer), stream));

    // fetch pointers to device data
    const at::Tensor vbo = at::reshape(vertex_buffer, {-1, 3}).cuda();
    const CUdeviceptr d_vertex_buffer = (CUdeviceptr)vbo.data_ptr<float>();
    const uint32_t num_vertices = vbo.size(0);
    const at::Tensor ibo = index_buffer.numel() ? at::reshape(index_buffer, {-1, 3}).cuda() : index_buffer;
    const CUdeviceptr d_index_buffer = (CUdeviceptr)(index_buffer.numel() ? ibo.data_ptr<int>() : nullptr);
    const uint32_t num_triangles = index_buffer.numel() ? ibo.size(0) : 0;

    // specify build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // specify build input, types and sizes
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = num_vertices;
    triangle_input.triangleArray.vertexBuffers = &d_vertex_buffer;
    if (index_buffer.numel()) {
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets = num_triangles;
        triangle_input.triangleArray.indexBuffer = d_index_buffer;
    }
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // query memory requirements for GAS
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK_THROW(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    // allocate device memory for scratch and output buffers
    CUdeviceptr d_gas_temp_buffer, d_AABB_buffer;
    CUDA_CHECK_THROW(cudaMallocAsync(reinterpret_cast<void**>(&d_gas_temp_buffer), gas_buffer_sizes.tempSizeInBytes, stream));
    CUDA_CHECK_THROW(cudaMallocAsync(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes, stream));
    CUDA_CHECK_THROW(cudaMallocAsync(reinterpret_cast<void**>(&d_AABB_buffer), 6*sizeof(float), stream));
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_AABBS;
    emitDesc.result = d_AABB_buffer;

    // trigger BVH build
    OPTIX_CHECK_THROW(optixAccelBuild(
        context,
        stream,
        &accel_options,
        &triangle_input,
        1,              // num build inputs
        d_gas_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &accel_handle,  // output handle
        &emitDesc,      // emitted property list
        1               // num emitted properties
    ));

    // store AABB in tensor and free scratch buffers
    AABB = at::clone(at::from_blob((void*)d_AABB_buffer, {2, 3}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA)));
    CUDA_CHECK_THROW(cudaFreeAsync(reinterpret_cast<void*>(d_gas_temp_buffer), stream));
    CUDA_CHECK_THROW(cudaFreeAsync(reinterpret_cast<void*>(d_AABB_buffer), stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> OptixContext::raytrace(at::Tensor ray_origins, at::Tensor ray_directions) {
    CHECK_TENSOR(ray_origins, 2, 3, at::kFloat);
    CHECK_TENSOR(ray_directions, 2, 3, at::kFloat);

    // allocate result tensors
    const auto n_rays = ray_origins.size(0);
    at::Tensor hit_positions = at::empty({n_rays, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    at::Tensor hit_normals = at::empty({n_rays, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    at::Tensor hit_primitiveIDs = at::empty({n_rays, 1}, at::device(at::kCUDA).dtype(at::kInt));

    // setup kernel params
    RaytraceParams params;
    params.handle = accel_handle;
    params.ray_origins = (float3*)ray_origins.contiguous().data_ptr<float>();
    params.ray_directions = (float3*)ray_directions.contiguous().data_ptr<float>();
    params.hit_positions = (float3*)hit_positions.data_ptr<float>();
    params.hit_normals = (float3*)hit_normals.data_ptr<float>();
    params.hit_primitiveIDs = hit_primitiveIDs.data_ptr<int>();

    // copy params to device and launch
    CUdeviceptr d_params;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK_THROW(cudaMallocAsync(reinterpret_cast<void**>(&d_params), sizeof(RaytraceParams), stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(d_params), reinterpret_cast<void*>(&params), sizeof(RaytraceParams), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK_THROW(optixLaunch(raytrace_pipeline, stream, d_params, sizeof(RaytraceParams), &raytrace_sbt, n_rays, 1, 1));
    CUDA_CHECK_THROW(cudaFreeAsync(reinterpret_cast<void*>(d_params), stream));

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return { hit_positions, hit_normals, hit_primitiveIDs };
}

at::Tensor OptixContext::visibility(at::Tensor ray_origins, at::Tensor ray_targets) {
    CHECK_TENSOR(ray_origins, 2, 3, at::kFloat);
    CHECK_TENSOR(ray_targets, 2, 3, at::kFloat);

    // allocate result tensors
    const auto n_rays = ray_origins.size(0);
    at::Tensor visibility = at::empty({n_rays, 1}, at::device(at::kCUDA).dtype(at::kFloat));

    // setup kernel params
    VisibilityParams params;
    params.handle = accel_handle;
    params.ray_origins = (float3*)ray_origins.contiguous().data_ptr<float>();
    params.ray_targets = (float3*)ray_targets.contiguous().data_ptr<float>();
    params.visibility = visibility.data_ptr<float>();

    // copy params to device and launch
    CUdeviceptr d_params;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK_THROW(cudaMallocAsync(reinterpret_cast<void**>(&d_params), sizeof(VisibilityParams), stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(d_params), reinterpret_cast<void*>(&params), sizeof(VisibilityParams), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK_THROW(optixLaunch(visibility_pipeline, stream, d_params, sizeof(VisibilityParams), &visibility_sbt, n_rays, 1, 1));
    CUDA_CHECK_THROW(cudaFreeAsync(reinterpret_cast<void*>(d_params), stream));

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return visibility;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> OptixContext::hitgen_cosine_power(at::Tensor origins, at::Tensor directions, float power, uint32_t seed, uint32_t n_hits) {
    CHECK_TENSOR(origins, 2, 3, at::kFloat);
    CHECK_TENSOR(directions, 2, 3, at::kFloat);

    // allocate result tensors
    at::Tensor hit_positions = at::zeros({n_hits, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    at::Tensor hit_normals = at::zeros({n_hits, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    at::Tensor hit_primitiveIDs = at::zeros({n_hits, 1}, at::device(at::kCUDA).dtype(at::kInt));
    at::Tensor counter = at::zeros({1}, at::device(at::kCUDA).dtype(at::kInt));

    // setup kernel params
    HitgenParams params;
    params.handle = accel_handle;
    params.origins = (float3*)origins.contiguous().data_ptr<float>();
    params.directions = (float3*)directions.contiguous().data_ptr<float>();
    params.power = power;
    params.seed = seed;
    params.counter = counter.data_ptr<int>();
    params.n_orgs = std::min(origins.size(0), directions.size(0));
    params.n_hits = n_hits;
    params.hit_positions = (float3*)hit_positions.data_ptr<float>();
    params.hit_normals = (float3*)hit_normals.data_ptr<float>();
    params.hit_primitiveIDs = (int*)hit_primitiveIDs.data_ptr<int>();

    // copy params to device and launch
    CUdeviceptr d_params;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK_THROW(cudaMallocAsync(reinterpret_cast<void**>(&d_params), sizeof(HitgenParams), stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(d_params), reinterpret_cast<void*>(&params), sizeof(HitgenParams), cudaMemcpyHostToDevice, stream));
    const size_t n_threads = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 1024;
    OPTIX_CHECK_THROW(optixLaunch(hitgen_pipeline, stream, d_params, sizeof(HitgenParams), &hitgen_sbt, n_threads, 1, 1));
    CUDA_CHECK_THROW(cudaFreeAsync(reinterpret_cast<void*>(d_params), stream));

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return { hit_positions, hit_normals, hit_primitiveIDs };
}

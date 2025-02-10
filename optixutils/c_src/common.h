#pragma once

#include <string>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <optix/optix.h>
#include <nvrtc.h>

#include <ATen/ATen.h>

// ------------------------------------------
// CUDA error check macros

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif

#ifndef STR
#define STR(x) STRINGIFY(x)
#endif

#ifndef FILE_LINE
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#endif

#ifndef CUDA_CHECK_PRINT
#define CUDA_CHECK_PRINT(x)                                                                                   \
    do {                                                                                                      \
        cudaError_t result = x;                                                                               \
        if (result != cudaSuccess)                                                                            \
            std::cerr << FILE_LINE ": " #x " failed with error " << cudaGetErrorString(result) << std::endl;   \
    } while(0)
#endif

#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(x)                                                                                              \
    do {                                                                                                                 \
        cudaError_t result = x;                                                                                          \
        if (result != cudaSuccess)                                                                                       \
            throw std::runtime_error(std::string(FILE_LINE ": " #x " failed with error ") + cudaGetErrorString(result));  \
    } while(0)
#endif

#ifndef OPTIX_CHECK_PRINT
#define OPTIX_CHECK_PRINT(x)                                                                                 \
    do {                                                                                                     \
        OptixResult res = x;                                                                                 \
        if (res != OPTIX_SUCCESS)                                                                            \
            std::cerr << std::string(FILE_LINE ": Optix call '" #x "' failed.") << std::endl;    \
    } while(0)
#endif

#ifndef OPTIX_CHECK_THROW
#define OPTIX_CHECK_THROW(x)                                                                                 \
    do {                                                                                                     \
        OptixResult res = x;                                                                                 \
        if (res != OPTIX_SUCCESS)                                                                            \
            throw std::runtime_error(std::string(FILE_LINE ": Optix call '" #x "' failed."));                            \
    } while(0)
#endif

#ifndef OPTIX_CHECK_THROW_LOG
#define OPTIX_CHECK_THROW_LOG(x)                                                                                                                          \
    do {                                                                                                                                                  \
        OptixResult res = x;                                                                                                                              \
        const size_t sizeof_log_returned = sizeof_log;                                                                                                    \
        sizeof_log = sizeof(log); /* reset sizeof_log for future calls */                                                                               \
        if (res != OPTIX_SUCCESS)                                                                                                                         \
            throw std::runtime_error(std::string(FILE_LINE ": Optix call '" #x "' failed. Log:\n") + log + (sizeof_log_returned == sizeof_log ? "" : "<truncated>")); \
    } while(0)
#endif

#ifndef NVRTC_CHECK_THROW
#define NVRTC_CHECK_THROW(x)                                                                                 \
    do {                                                                                                     \
        nvrtcResult res = x;                                                                                 \
        if (res != NVRTC_SUCCESS)                                                                            \
            throw std::runtime_error(std::string(FILE_LINE ": Nvrtc call '" #x "' failed: " + std::string(nvrtcGetErrorString(res))));                            \
    } while(0)
#endif

// ------------------------------------------
// ATen error check macros

#define CHECK_TENSOR(X, DIMS, CHANNELS, TYPE) \
    AT_ASSERTM(X.is_cuda(), #X " must be a cuda tensor"); \
    AT_ASSERTM(X.scalar_type() == TYPE, #X " must be " #TYPE " tensor"); \
    AT_ASSERTM(X.dim() == DIMS, #X " must have " #DIMS " dimensions"); \
    AT_ASSERTM(X.size(std::max(0, DIMS - 1)) == CHANNELS, #X " must have " #CHANNELS " channels")

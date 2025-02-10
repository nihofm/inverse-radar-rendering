import os
import torch

# set device capability
device_capability = torch.cuda.get_device_capability()
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{device_capability[0]}.{device_capability[1]}'
print(f'{os.path.basename(os.path.dirname(__file__))}: compiling with TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}...')

# --------------------------------------------------------------
# compile OptixContext as torch extension

directory = os.path.dirname(__file__)

source_files = [
    os.path.join(directory, 'c_src/bindings.cpp'),
    os.path.join(directory, 'c_src/context.cpp'),
]
include_paths = [
    os.path.join(directory, 'include'),
    os.path.join(directory, 'c_src'),
]

ldflags = [
    '-lnvrtc'
]

cflags = []

# Slightly different handling for Windows
if os.name == 'nt':
    directory = directory.replace('\\', '/')
    ldflags = [
        'nvrtc.lib',
        'Advapi32.lib',
    ]
    # Windows defines min and max macros which conflict with std::min and std::max
    cflags.append('-DNOMINMAX') # disable min and max macros

cflags.append(f'-DEXTENSION_BASE_DIRECTORY="{directory}"')

optixutils = torch.utils.cpp_extension.load('optixutils', source_files, extra_include_paths=include_paths, extra_cflags=cflags, extra_ldflags=ldflags, with_cuda=True)

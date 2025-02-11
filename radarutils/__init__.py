import os
import torch
import slangtorch

# set device capability
device_capability = torch.cuda.get_device_capability()
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{device_capability[0]}.{device_capability[1]}'
print(f'{os.path.basename(os.path.dirname(__file__))}: compiling with TORCH_CUDA_ARCH_LIST: {os.environ["TORCH_CUDA_ARCH_LIST"]}...')

# --------------------------------------------------------------
# Torch autograd function for differentiable signal generation w/ differentiable material

class SFCWSignalMaterialFunc(torch.autograd.Function):
    module = None

    @staticmethod
    def set_defines(material_type: int, use_apc: bool):
        SFCWSignalMaterialFunc.module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "sfcw_signal_material.slang"), defines={'MATERIAL_TYPE': material_type, 'USE_APC': int(use_apc)})

    @staticmethod
    def forward(ctx, tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_normals, hit_material):
        assert SFCWSignalMaterialFunc.module, "call SFCWSignalMaterialFunc.set_defines first!"
        ctx.save_for_backward(tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_normals, hit_material)
        return SFCWSignalMaterialFunc.module.sfcw_signal_material_fwd(tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_normals, hit_material)

    @staticmethod
    def backward(ctx, output_grad):
        assert SFCWSignalMaterialFunc.module, "call SFCWSignalMaterialFunc.set_defines first!"
        [tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_normals, hit_material] = ctx.saved_tensors
        hit_positions_grad, hit_normals_grad, hit_material_grad = SFCWSignalMaterialFunc.module.sfcw_signal_material_bwd(tx_positions, rx_positions, frequencies, tx_pattern, rx_pattern, hit_positions, hit_normals, hit_material, output_grad.contiguous())
        return None, None, None, None, None, hit_positions_grad, hit_normals_grad, hit_material_grad

# --------------------------------------------------------------
# Torch autograd function for differentiable SFCW radar reconstruction

class SFCWRecoFunc(torch.autograd.Function):
    module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "sfcw_reco.slang"))

    @staticmethod
    def forward(ctx, tx_positions, rx_positions, frequencies, signal, AABB, n_voxels_xy, n_voxels_z):
        ctx.save_for_backward(tx_positions, rx_positions, frequencies, signal, AABB)
        return SFCWRecoFunc.module.sfcw_reco_AABB_fwd(tx_positions, rx_positions, frequencies, signal, AABB, n_voxels_xy, n_voxels_z)

    @staticmethod
    def backward(ctx, output_grad):
        [tx_positions, rx_positions, frequencies, signal, AABB] = ctx.saved_tensors
        signal_grad = SFCWRecoFunc.module.sfcw_reco_AABB_bwd(tx_positions, rx_positions, frequencies, signal, AABB, output_grad.contiguous())
        return None, None, None, signal_grad, None, None, None

# --------------------------------------------------------------
# Torch autograd function for trilinear grid sampling

class GridLookupFunc(torch.autograd.Function):
    module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "grid_lookup.slang"))

    @staticmethod
    def forward(ctx, ipos, grid):
        ctx.save_for_backward(ipos, grid)
        return GridLookupFunc.module.grid_lookup_fwd(ipos, grid)

    @staticmethod
    def backward(ctx, output_grad):
        [ipos, grid] = ctx.saved_tensors
        grid_grad = GridLookupFunc.module.grid_lookup_bwd(ipos, grid, output_grad)
        return None, grid_grad

# --------------------------------------------------------------
# Torch autograd function for barycentric mesh sampling

class MeshLookupFunc(torch.autograd.Function):
    module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "mesh_lookup.slang"))

    @staticmethod
    def forward(ctx, hit_positions, hit_primIDs, vbo, ibo, features):
        ctx.save_for_backward(hit_positions, hit_primIDs, vbo, ibo, features)
        return MeshLookupFunc.module.mesh_lookup_fwd(hit_positions, hit_primIDs, vbo, ibo, features)

    @staticmethod
    def backward(ctx, output_grad):
        [hit_positions, hit_primIDs, vbo, ibo, features] = ctx.saved_tensors
        features_grad = MeshLookupFunc.module.mesh_lookup_bwd(hit_positions, hit_primIDs, vbo, ibo, features, output_grad)
        return None, None, None, None, features_grad

# --------------------------------------------------------------
# Torch autograd function for regularization after mesh sampling

class MeshRegularizeFunc(torch.autograd.Function):
    module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "mesh_regularize.slang"))

    @staticmethod
    def forward(ctx, vbo, start_end, neighbors, features):
        ctx.save_for_backward(vbo, start_end, neighbors, features)
        return MeshRegularizeFunc.module.mesh_regularize_fwd(vbo, start_end, neighbors, features)

    @staticmethod
    def backward(ctx, output_grad):
        [vbo, start_end, neighbors, features] = ctx.saved_tensors
        features_grad = MeshRegularizeFunc.module.mesh_regularize_bwd(vbo, start_end, neighbors, features, output_grad)
        return None, None, None, features_grad

# ---------------------------------------------------
# Torch autograd function for normalmapping

class NormalmapFunction(torch.autograd.Function):
    module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "normalmap.slang"))

    @staticmethod
    def forward(ctx, normals: torch.Tensor, normalmap: torch.Tensor):
        ctx.save_for_backward(normals, normalmap)
        return NormalmapFunction.module.normalmap_fwd(normals, normalmap)

    @staticmethod
    def backward(ctx, output_grad):
        [normals, normalmap] = ctx.saved_tensors
        return NormalmapFunction.module.normalmap_bwd(normals, normalmap, output_grad)

# ---------------------------------------------------
# Ray generation utility functions (non-differentiable)

_raygen_module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "raygen.slang"))

def raygen_pinhole(
        camera_positions:   torch.tensor,   # [N_CAMERAS, 3]
        camera_directions:  torch.tensor,   # [N_CAMERAS, 3]
        camera_fovy:        torch.tensor,   # [N_CAMERAS, 1]
        width:              int,
        height:             int):
    out_pos = torch.empty((camera_positions.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    out_dir = torch.empty((camera_positions.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    _raygen_module.raygen_pinhole_camera(
        cam_pos=camera_positions,
        cam_dir=camera_directions,
        cam_fovy=camera_fovy,
        out_pos=out_pos,
        out_dir=out_dir
    ).launchRaw(blockSize=(1, 32, 32), gridSize=(camera_positions.size(0), (height+31)//32, (width+31)//32))
    return out_pos, out_dir

def raygen_random_aabb(
        origins:            torch.tensor,   # [N_BATCH, 3]
        AABB:               torch.tensor,   # [2, 3]
        seed:               int,
        width:              int,
        height:             int):
    out_pos = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    out_dir = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    _raygen_module.raygen_random_aabb(
        origins=origins,
        AABB=AABB,
        seed=seed,
        out_pos=out_pos,
        out_dir=out_dir
    ).launchRaw(blockSize=(1, 32, 32), gridSize=(out_pos.size(0), (height+31)//32, (width+31)//32))
    return out_pos, out_dir

def raygen_uniform_hemisphere(
        origins:            torch.tensor,   # [N_BATCH, 3]
        normals:            torch.tensor,   # [N_BATCH, 3]
        seed:               int,
        width:              int,
        height:             int):
    out_pos = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    out_dir = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    _raygen_module.raygen_uniform_hemisphere(
        origins=origins,
        normals=normals,
        seed=seed,
        out_pos=out_pos,
        out_dir=out_dir
    ).launchRaw(blockSize=(1, 32, 32), gridSize=(out_pos.size(0), (height+31)//32, (width+31)//32))
    return out_pos, out_dir

def raygen_cosine_hemisphere(
        origins:            torch.tensor,   # [N_BATCH, 3]
        normals:            torch.tensor,   # [N_BATCH, 3]
        seed:               int,
        width:              int,
        height:             int):
    out_pos = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    out_dir = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    _raygen_module.raygen_cosine_hemisphere(
        origins=origins,
        normals=normals,
        seed=seed,
        out_pos=out_pos,
        out_dir=out_dir
    ).launchRaw(blockSize=(1, 32, 32), gridSize=(out_pos.size(0), (height+31)//32, (width+31)//32))
    return out_pos, out_dir

def raygen_cosine_power_hemisphere(
        origins:            torch.tensor,   # [N_BATCH, 3]
        normals:            torch.tensor,   # [N_BATCH, 3]
        power:              torch.tensor,   # [N_BATCH, 1]
        seed:               int,
        width:              int,
        height:             int):
    out_pos = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    out_dir = torch.empty((origins.size(0), height, width, 3), dtype=torch.float32, device='cuda')
    _raygen_module.raygen_cosine_power_hemisphere(
        origins=origins,
        normals=normals,
        power=power,
        seed=seed,
        out_pos=out_pos,
        out_dir=out_dir
    ).launchRaw(blockSize=(1, 32, 32), gridSize=(out_pos.size(0), (height+31)//32, (width+31)//32))
    return out_pos, out_dir

# --------------------------------------------------------------
# Raymarching SFCW radar reconstruction for visualization (non-differentiable)

_raymarch_module = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "slang", "sfcw_raymarch.slang"))

def sfcw_signal_raymarch_reco(
        cam_pos:        torch.tensor,   # [N_BATCH, 3]
        cam_dir:        torch.tensor,   # [N_BATCH, 3]
        cam_fovy:       torch.tensor,   # [N_BATCH, 1]
        AABB:           torch.tensor,   # [2, 3]
        reco:           torch.tensor,   # [N_VOXELS, N_VOXELS, N_VOXELS]
        width:          int = 1920,     # [1]
        height:         int = 1080,     # [1]
        seed:           int = 42):      # [1]
    result = torch.empty((cam_pos.size(0), height, width, 2), dtype=torch.float32, device='cuda')
    _raymarch_module.sfcw_raymarch_kernel(
        cam_pos=cam_pos,
        cam_dir=cam_dir,
        cam_fovy=cam_fovy,
        AABB=AABB,
        reco=reco,
        seed=seed,
        result=result
    ).launchRaw(blockSize=(1, 32, 32), gridSize=(cam_pos.size(0), (height+31)//32, (width+31)//32))
    intensity, depth = result[..., 0], result[..., 1]
    return intensity, depth

import torch
import trimesh
import optixutils
import radarutils
import numpy as np
from matplotlib import cm
from .encoding import MultiResHashGrid

# helper to apply a matplotlib colormap
def _colormap(v, colormap):
    cmap_numpy = np.apply_along_axis(colormap, 0, torch.clamp(v, 0.0, 1.0).flatten().detach().cpu().numpy()) # convert v to colormap
    if v.dim() == 2:
        cmap_numpy = cmap_numpy.reshape(v.size(0), v.size(1), 4)[..., 0:3]
    elif v.dim() ==3 :
        cmap_numpy = cmap_numpy.reshape(v.size(0), v.size(1), v.size(2), 4)[..., 0:3]
    else:
        raise RuntimeError("kaputt")
    return torch.from_numpy(np.squeeze(cmap_numpy)).to(v.device)

# --------------------------------------------------------------
# Different material options

class GlobalMaterial(torch.nn.Module):
    def __init__(self, n_params: int) -> None:
        super().__init__()
        self.params = torch.nn.Parameter(0.01 * torch.randn(1, n_params).cuda())

    def forward(self, ipos: torch.Tensor):
        return torch.sigmoid(self.params).expand(ipos.size(0), self.params.size(1)).clone()

class HashGridMaterial(torch.nn.Module):
    def __init__(self, n_params: int, n_levels: int = 16, n_features_per_level: int = 4, log2_hashmap_size: int = 18) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            MultiResHashGrid(3, n_levels=n_levels, n_features_per_level=n_features_per_level, log2_hashmap_size=log2_hashmap_size),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, n_params), torch.nn.Sigmoid(),
        ).cuda()

    def forward(self, ipos: torch.Tensor):
        return self.model(ipos)

class VoxelGridMaterial(torch.nn.Module):
    def __init__(self, n_params: int, n_voxels: int) -> None:
        super().__init__()
        self.grid = torch.nn.Parameter(0.01 * torch.randn(n_voxels, n_voxels, n_voxels, n_params, dtype=torch.float32).cuda())

    def forward(self, ipos: torch.Tensor):
        return radarutils.GridLookupFunc.apply(ipos, self.grid)

class VertexMaterial(torch.nn.Module):
    def __init__(self, n_params: int, n_vertices: int) -> None:
        super().__init__()
        self.features = torch.nn.Parameter(0.01 * torch.randn(n_vertices, n_params, dtype=torch.float32).cuda())

    def forward(self, hit_positions: torch.Tensor, hit_primIDs: torch.Tensor, vbo: torch.Tensor, ibo: torch.Tensor):
        return radarutils.MeshLookupFunc.apply(hit_positions, hit_primIDs, vbo, ibo, self.features) # sigmoid activated features

class VertexNormalmap(torch.nn.Module):
    def __init__(self, n_vertices: int) -> None:
        super().__init__()
        self.features = torch.nn.Parameter(torch.zeros(n_vertices, 3, dtype=torch.float32).cuda())

    def forward(self, hit_positions: torch.Tensor, hit_normals: torch.Tensor, hit_primIDs: torch.Tensor, vbo: torch.Tensor, ibo: torch.Tensor):
        normalmap = radarutils.MeshLookupFunc.apply(hit_positions, hit_primIDs, vbo, ibo, self.features) # sigmoid activated features
        N = hit_normals + (normalmap * 2 - 1) * 0.25 # re-map to vector offset in [-0.25, +0.25]
        return N / torch.linalg.vector_norm(N, dim=-1, keepdim=True)

# --------------------------------------------------------------
# Differentiable radar w.r.t. material

class DiffRadarMaterial(torch.nn.Module):
    def __init__(self,
                 tx_positions: torch.Tensor,    # [n_tx, 3]
                 rx_positions: torch.Tensor,    # [n_rx, 3]
                 frequencies: torch.Tensor,     # [n_freq, 1]
                 tx_pattern: torch.Tensor,      # [n_theta, n_phi, 5] -> (dBi_abs, dBi_s, phase_s, dBi_p, phase_p)
                 rx_pattern: torch.Tensor,      # [n_theta, n_phi, 5] -> (dBi_abs, dBi_s, phase_s, dBi_p, phase_p)
                 mesh_vertices: torch.Tensor,   # [n_vertices, 3]
                 mesh_indices: torch.Tensor,    # [n_indices, 3] -> triangles only
                 material_type: int,            # select material model (see switch case in sfcw_signal_material.slang)
                 use_normalmap: bool,           # select if VertexNormalmap should be used
                 use_reg_offset: bool,          # select if fine registration should be optimized
                 use_apc: bool,                 # select if antenna power characteristic should be considered
                 material_storage: str,         # select material storage option (see code above)
                 device=torch.cuda.current_device()) -> None:
        super().__init__()
        # setup radar constants
        self.tx_positions = tx_positions.to(device)
        self.rx_positions = rx_positions.to(device)
        self.frequencies = frequencies.to(device)
        self.tx_pattern = tx_pattern.to(device)
        self.rx_pattern = rx_pattern.to(device)
        self.radar_dir = torch.tensor([0, 0, 1], dtype=torch.float32, device=device) # convention: radar always looking in +Z
        # residual gain parameter (power gain), empirical initialization to 6dB
        self.gain_dB = torch.nn.Parameter(torch.tensor([6.0]).to(device))
        # global offset for registration correction
        self.reg_offset = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).to(device)) if use_reg_offset else None
        # setup mesh data
        self.vbo = mesh_vertices.to(torch.float32).to(device)
        self.ibo = mesh_indices.to(torch.int32).to(device)
        if material_storage == 'vertex':
            # compute flattened list of vertex neighbors and their start/end indices for regularization
            mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_indices)
            n_neighbors = np.array([len(vn) for vn in mesh.vertex_neighbors])
            if max(n_neighbors) > 32:
                raise RuntimeError(f"Exceeding supported max mesh neighbors of 32 ({max(n_neighbors)}), hardcoded as Slang [MaxIters()]!")
            self.start_end = torch.from_numpy(np.stack((np.cumsum(np.pad(n_neighbors[:-1], (1, 0))), np.cumsum(n_neighbors)), axis=-1)).to(dtype=torch.int32, device=device)
            self.neighbors = torch.from_numpy(np.array([n for vn in mesh.vertex_neighbors for n in vn])).to(dtype=torch.int32, device=device)
        # setup OptiX
        self.optix_ctx = optixutils.OptixContext()
        self.optix_ctx.build_bvh(self.vbo, self.ibo)
        self.AABB = torch.stack((torch.min(self.vbo, dim=0).values, torch.max(self.vbo, dim=0).values))
        print("AABB min:", self.AABB[0].cpu().numpy())
        print("AABB max:", self.AABB[1].cpu().numpy())
        # setup material type and storage
        radarutils.SFCWSignalMaterialFunc.set_defines(material_type, use_apc)
        if material_storage == 'global':
            # use one global set of parameters
            self.material = GlobalMaterial(8).to(device)
        elif material_storage == 'voxelgrid':
            # use regular voxel grid
            self.material = VoxelGridMaterial(8, 32).to(device)
        elif material_storage == 'hashgrid':
            # use hash grid + MLP
            self.material = HashGridMaterial(8).to(device)
        elif material_storage == 'vertex':
            # use per vertex parameters
            self.material = VertexMaterial(8, self.vbo.size(0)).to(device)
        else:
            raise RuntimeError("Invalid material storage type!")
        # setup normalmap
        self.normalmap = VertexNormalmap(self.vbo.size(0)).to(device) if use_normalmap else None

    def forward(self, seed: int, n_hits: int) -> torch.Tensor:
        # raytrace hits
        hit_positions, hit_normals, hit_primIDs = self.optix_ctx.hitgen_cosine_power(self.tx_positions, self.radar_dir.expand_as(self.tx_positions), 5.0, seed, n_hits)
        # evaluate material
        if isinstance(self.material, VertexMaterial):
            hit_material = self.material(hit_positions, hit_primIDs, self.vbo, self.ibo)
        else:
            hit_material = self.material((hit_positions - self.AABB[0]) / (self.AABB[1] - self.AABB[0]))
        # apply normalmap (optional) and registration offset
        if self.normalmap:
            hit_normals = self.normalmap(hit_positions, hit_normals, hit_primIDs, self.vbo, self.ibo)
        hit_positions = self.apply_registration_offset(hit_positions)
        # generate radar signal
        signal = radarutils.SFCWSignalMaterialFunc.apply(
            self.tx_positions,
            self.rx_positions,
            self.frequencies,
            self.tx_pattern,
            self.rx_pattern,
            hit_positions,
            hit_normals,
            hit_material,
        )
        # apply power gain (G_t = G_r = G**2) and normalize signal
        return signal * (torch.pow(10, self.gain_dB / 10.0)**2 / hit_positions.size(0))

    # apply registration offset to hit positions, maximum is half of the max wavelength in each direction
    def apply_registration_offset(self, hit_positions):
        if self.reg_offset is None: return hit_positions
        wavelength = 299792458 / self.frequencies[0] # use lowest frequency -> longest wavelength
        return hit_positions + (torch.sigmoid(self.reg_offset) - 0.5) * wavelength

    # apply regularization based on material type
    def regularization_loss(self):
        if isinstance(self.material, VertexMaterial):
            return torch.mean(radarutils.MeshRegularizeFunc.apply(self.vbo, self.start_end, self.neighbors, self.material.features))
        else:
            return torch.tensor([0.0], device=self.vbo.device)

    # helper to compute depth resolution of radar image in meters (~1.4cm)
    def depth_resolution(self):
        bandwidth = self.frequencies[-1] - self.frequencies[0]
        return 299792458 / (2 * bandwidth)

    # helper to compute max unambigious range in meters (~1.9m)
    def max_unambigious_range(self):
        return self.depth_resolution() * len(self.frequencies)

    # helper to create an image reconstruction from the raw radar signal
    def reco_from_signal(self, signal: torch.Tensor, n_voxels: int) -> torch.Tensor:
        n_voxels_z = max(2, int((self.AABB[1, 2] - self.AABB[0, 2]) / self.depth_resolution()) * 2)
        reco = radarutils.SFCWRecoFunc.apply(self.tx_positions, self.rx_positions, self.frequencies, signal, self.AABB, n_voxels, n_voxels_z)
        return torch.abs(torch.view_as_complex(reco))

    # helper to visualize geometry
    def visualize_depth(self, cam_pos=[0, 0, 0], cam_dir=[0, 0, 1], width=1024, height=1024, fovy=40):
        with torch.no_grad():
            cam_pos = torch.tensor([cam_pos], dtype=torch.float32, device='cuda')
            cam_dir = torch.tensor([cam_dir], dtype=torch.float32, device='cuda')
            cam_fovy = torch.zeros_like(cam_pos[..., 0:1]) + fovy
            org, dir = radarutils.raygen_pinhole(cam_pos, cam_dir, cam_fovy, width, height)
            hit_positions, _, primIDs = self.optix_ctx.raytrace(org.reshape(-1, 3), dir. reshape(-1, 3))
            hit_positions = self.apply_registration_offset(hit_positions)
            distance = torch.linalg.vector_norm(hit_positions - cam_pos, dim=-1, keepdim=True)
            min_depth = torch.min(torch.masked_select(distance, primIDs != 0xFFFFFFFF))
            max_depth = torch.max(torch.masked_select(distance, primIDs != 0xFFFFFFFF))
            distance = distance.reshape(height, width, 1).expand(-1, -1, 3)
            return ((distance - min_depth) / (max_depth - min_depth)).transpose(-3, -1).transpose(-2, -1)

    # helper to visualize normals
    def visualize_normals(self, cam_pos=[0, 0, 0], cam_dir=[0, 0, 1], width=1024, height=1024, fovy=40):
        with torch.no_grad():
            cam_pos = torch.tensor([cam_pos], dtype=torch.float32, device='cuda')
            cam_dir = torch.tensor([cam_dir], dtype=torch.float32, device='cuda')
            cam_fovy = torch.zeros_like(cam_pos[..., 0:1]) + fovy
            org, dir = radarutils.raygen_pinhole(cam_pos, cam_dir, cam_fovy, width, height)
            hit_positions, hit_normals, hit_primIDs = self.optix_ctx.raytrace(org.reshape(-1, 3), dir.reshape(-1, 3))
            if self.normalmap:
                hit_normals = self.normalmap(hit_positions, hit_normals, hit_primIDs, self.vbo, self.ibo)
            return torch.abs(hit_normals).reshape(height, width, 3).transpose(-3, -1).transpose(-2, -1)

    # helper to visualize materials
    def visualize_material(self, cam_pos=[0, 0, 0], cam_dir=[0, 0, 1], width=1024, height=1024, fovy=40):
        with torch.no_grad():
            cam_pos = torch.tensor([cam_pos], dtype=torch.float32, device='cuda')
            cam_dir = torch.tensor([cam_dir], dtype=torch.float32, device='cuda')
            cam_fovy = torch.zeros_like(cam_pos[..., 0:1]) + fovy
            org, dir = radarutils.raygen_pinhole(cam_pos, cam_dir, cam_fovy, width, height)
            hit_positions, _, primIDs = self.optix_ctx.raytrace(org.reshape(-1, 3), dir. reshape(-1, 3))
            if isinstance(self.material, VertexMaterial):
                hit_material = self.material(hit_positions, primIDs, self.vbo, self.ibo)
            else:
                hit_material = self.material((hit_positions - self.AABB[0]) / (self.AABB[1] - self.AABB[0]))
            hit_material = torch.where(primIDs != -1, hit_material, torch.zeros_like(hit_material))
            hit_material = hit_material.reshape(height, width, hit_material.size(-1)).transpose(-3, -1).transpose(-2, -1)
            return _colormap(hit_material, cm.magma).transpose(-3, -1).transpose(-2, -1)

    # helper to visualize a reco volume
    def visualize_reco(self, reco, threshold_db, cam_pos=[0, 0, 0], cam_dir=[0, 0, 1], width=1024, height=1024, fovy=40):
        with torch.no_grad():
            cam_pos = torch.tensor([cam_pos], dtype=torch.float32, device=reco.device)
            cam_dir = torch.tensor([cam_dir], dtype=torch.float32, device=reco.device)
            cam_fovy = torch.zeros_like(cam_pos[..., 0:1]) + fovy
            view_reco, _ = radarutils.sfcw_signal_raymarch_reco(cam_pos, cam_dir, cam_fovy, self.AABB, reco, width, height)
            view_reco = 20 * torch.log10(view_reco[0] / torch.max(view_reco[0]))        # convert to dB
            view_reco = torch.clamp(view_reco + threshold_db, min=0.0) / threshold_db   # map from [-args['threshold_db'] to 0db]
            return _colormap(view_reco, cm.viridis).transpose(-3, -1).transpose(-2, -1)

    # helper to visualize difference of two reco volumes
    def visualize_reco_diff(self, reco_pred, reco_target, threshold_db, cam_pos=[0, 0, 0], cam_dir=[0, 0, 1], width=1024, height=1024, fovy=40):
        with torch.no_grad():
            cam_pos = torch.tensor([cam_pos], dtype=torch.float32, device=reco_target.device)
            cam_dir = torch.tensor([cam_dir], dtype=torch.float32, device=reco_target.device)
            cam_fovy = torch.zeros_like(cam_pos[..., 0:1]) + fovy
            view_reco_diff, _ = radarutils.sfcw_signal_raymarch_reco(cam_pos, cam_dir, cam_fovy, self.AABB, torch.abs(reco_pred - reco_target), width, height)
            view_reco_diff = 20 * torch.log10(view_reco_diff[0] / torch.max(reco_target)) # convert to dB
            view_reco_diff = torch.clamp(view_reco_diff + threshold_db, min=0.0) / threshold_db   # map from [-args['threshold_db'] to 0db]
            return _colormap(view_reco_diff, cm.turbo).transpose(-3, -1).transpose(-2, -1)

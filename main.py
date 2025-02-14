import os
import time
import json
import trimesh
import argparse
import datetime
import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from radarutils.diffradar import DiffRadarMaterial

# -----------------------------------------------------------
# Radar helpers

def to_spherical(xyz):
    # convention: theta in [0, 2*PI) and phi in [0, PI]
    theta = np.arccos(np.clip(xyz[:, 2] / np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2), -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi

def lookup_antenna_pattern(xyz, pattern) -> np.ndarray:
    theta, phi = to_spherical(xyz)
    x = (pattern.shape[0] * phi / np.pi).astype(np.int32) % pattern.shape[0]
    y = (pattern.shape[1] * theta / (2 * np.pi)).astype(np.int32) % pattern.shape[1]
    return pattern[x, y]

def load_antenna_pattern_cst(filename) -> np.ndarray:
    data    = np.loadtxt(filename, skiprows=2).astype(np.float32)   # [rows, cols] layout
    n_theta = np.unique(data[:, 0]).size                            # input: degrees in [-180, +179]
    n_phi   = np.unique(data[:, 1]).size                            # input: degrees in [-90, +90]
    data[:, 0] = np.deg2rad(data[:, 0])                             # theta: convert degrees to radians and map to [0, 2PI)
    data[:, 1] = np.deg2rad(data[:, 1])                             # phi: convert degrees to radians and map to [0, PI]
    data[:, 2] = np.power(10, data[:, 2] / 10)                      # abs_power_dB: convert dB to linear
    data = data.reshape(n_phi, n_theta, -1)                         # reshape to [theta, phi, values]
    data = np.roll(data, shift=(n_phi//2, n_theta//2), axis=(0, 1)) # rotate values to match [0, PI], [0, 2*PI) convention
    print(f'antenna data layout: {data.shape}, {data.dtype}')
    return data[:, :, 2:3] # return layout:[n_phi, n_theta, [power]]

# -----------------------------------------------------------
# Main script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_or_config', type=str, help="Path to dataset directory or config file to load")
    parser.add_argument('--dataset', default=None, type=str, help="Path to dataset")
    parser.add_argument('--name', default=None, type=str, help="Name of this run")
    parser.add_argument('--n_hits', default=1<<16, type=int, help="Number of hit points to use for signal generation")
    parser.add_argument('--seed', default=np.random.randint(2**31-1), type=int, help="Random seed for pseudorandom number generation")
    parser.add_argument('--n_voxels', default=128, type=int, help="Number of voxels per side for holographic reconstruction")
    parser.add_argument('--threshold_db', default=35, type=float, help="Dynamic range in dB when visualizing the reconstruction")
    parser.add_argument('--n_epochs', default=250, type=int, help="Number of inverse reconstruction iterations")
    parser.add_argument('--lr', default=1e-1, type=float, help="Learning rate during reconstruction")
    parser.add_argument('--loss', default='l1_complex', type=str, choices=['l1', 'l1_complex', 'l2', 'l2_reco'], help="Select loss function")
    parser.add_argument('--device', default='cuda:0', type=str, help="CUDA device to compute stuff on")
    parser.add_argument('--material', default=4, type=int, help="Which material model to use (0: baseline, 1: mixed phong, 2: layered BRDF, 3: fresnel smooth, 4: fresnel rough)")
    parser.add_argument('--material_storage', default='voxelgrid', type=str, choices=['global', 'voxelgrid', 'hashgrid', 'vertex'], help="Which material datastructure to use")
    parser.add_argument('--no_emptyfiltered', dest='use_emptyfiltered', default=True, action='store_false', help="Avoid emptyfiltered signal and use calibrated raw data")
    parser.add_argument('--no_reg_offset', dest='use_reg_offset', default=True, action='store_false', help="Do not optimize fine registration alongside material params")
    parser.add_argument('--use_normalmap', default=False, action='store_true', help="Optimize normalmap alongside material params")
    parser.add_argument('--use_apc', default=False, action='store_true', help="Consider antenna power characteristic during signal computation")
    parser.add_argument('--no_zeroed', dest='use_zeroed', default=True, action='store_false', help="Zeroize simulated antenna pairs where real signal is zeroed")
    parser.add_argument('--synthetic', default=False, action='store_true', help="Use syntethic radar dataset for gradient validation")
    parser.add_argument('--weights', default=None, type=str, help="Path to weights to load before training")
    # parse and print cmdline args
    args = vars(parser.parse_args())
    # resolve 'dataset_or_config' to 'dataset'
    if args['dataset_or_config'].lower().endswith('.json'):
        config = json.load(open(args['dataset_or_config'], 'r'))['config']
        for key in config:
            if key not in ['dataset_or_config', 'inference']:
                args[key] = config[key]
    else:
        args['dataset'] = args['dataset_or_config']
        if args['name'] is None:
            args['name'] = os.path.basename(os.path.dirname(os.path.abspath(args['dataset'])))
        args['name'] += '-' + os.urandom(4).hex() # avoid name clashes
    print('Config:')
    for key in args:
        print(f'\t{key}: {args[key]}')
    print(f'Device: {torch.cuda.get_device_properties(args["device"])}')

    # set random seeds
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    # torch.autograd.set_detect_anomaly(True) # use for debugging

    # MAROON path handling
    base_path = os.path.abspath(args['dataset'])
    alignment_path = os.path.join(base_path, 'alignment.json')
    mesh_path_smooth = os.path.join(base_path, 'photogrammetry', 'mesh_smoothed.obj')
    mesh_path_smooth_mask = os.path.join(base_path, 'photogrammetry', 'mesh_masked_smoothed.obj')
    mesh_path = mesh_path_smooth_mask if os.path.exists(mesh_path_smooth_mask) else mesh_path_smooth
    rgb_path = os.path.join(base_path, 'photogrammetry', 'rgb')
    radar_path = os.path.join(base_path, 'radar_72.0_82.0_128', 'calibrated_data')
    Tx_path = os.path.join(os.path.dirname(__file__), 'submodules', 'maroon', 'data', 'qar50sc', 'Tx.dat')
    Rx_path = os.path.join(os.path.dirname(__file__), 'submodules', 'maroon', 'data', 'qar50sc', 'Rx.dat')
    Tx_pattern_path = os.path.join(os.path.dirname(__file__), 'data', 'Tx_pattern.txt')
    Rx_pattern_path = os.path.join(os.path.dirname(__file__), 'data', 'Rx_pattern.txt')

    # --------------------------------------------------------------
    # Scene setup

    # load mesh
    print(f'Loading mesh from: {mesh_path}...')
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices, indices = torch.from_numpy(mesh.vertices), torch.from_numpy(mesh.faces)
    # align mesh with radar
    mesh_transform = torch.from_numpy(np.array(json.load(open(alignment_path, 'r'))['photogrammetry2radar']))
    vertices = torch.matmul(torch.cat([vertices, torch.ones_like(vertices[:, 0:1])], dim=-1), mesh_transform.T)[:, 0:3]
    vertices[..., 0] *= -1 # flip x-axis to align with mesh
    print(f'Mesh: #vertices: {vertices.size(0)}, #indices: {indices.size(0)}')

    # --------------------------------------------------------------
    # Radar setup

    # radar centered at origin looking at +Z
    radar_pos = torch.tensor([0, 0, 0], dtype=torch.float32, device=args['device'])
    radar_dir = torch.tensor([0, 0, 1], dtype=torch.float32, device=args['device'])
    # load Tx and Rx positions from file
    tx_positions = torch.from_numpy(np.loadtxt(Tx_path, skiprows=1, delimiter=",")[1:95]).to(dtype=torch.float32, device=args['device'])
    rx_positions = torch.from_numpy(np.loadtxt(Rx_path, skiprows=1, delimiter=",")[1:95]).to(dtype=torch.float32, device=args['device'])
    # load Tx and Rx antenna pattern from CST
    if not os.path.exists(Tx_pattern_path): print(f'No TX antenna pattern specified, using identity...')
    if not os.path.exists(Rx_pattern_path): print(f'No RX antenna pattern specified, using identity...')
    tx_pattern = torch.from_numpy(load_antenna_pattern_cst(Tx_pattern_path)) if os.path.exists(Tx_pattern_path) else torch.tensor([1], dtype=torch.float32, device=args['device'])
    rx_pattern = torch.from_numpy(load_antenna_pattern_cst(Rx_pattern_path)) if os.path.exists(Rx_pattern_path) else torch.tensor([1], dtype=torch.float32, device=args['device'])
    if args['use_apc'] and (not os.path.exists(Tx_pattern_path) or not os.path.exists(Rx_pattern_path)):
        raise RuntimeError(f'--use_apc requires both TX and RX antenna pattern files')
    # load raw radar data and frequencies
    print(f'Loading radar data from: {radar_path}...')
    radar_files = [os.path.join(radar_path, f) for f in os.listdir(radar_path) if '.npy' in f]
    if args['use_emptyfiltered']:
        radar_files = list(sorted(filter(lambda x: 'emptyfiltered' in x, radar_files)))
    else:
        radar_files = list(sorted(filter(lambda x: 'emptyfiltered' not in x, radar_files)))
    print(f'Found {len(radar_files)} {"emptyfiltered" if args["use_emptyfiltered"] else "calibrated"} measurements...')
    radar_data_raw = [np.load(radar_file, allow_pickle=True) for radar_file in radar_files]
    radar_frequencies = torch.stack([torch.from_numpy(radar_data.item().get('fvec')).to(dtype=torch.float32, device=args['device']) for radar_data in radar_data_raw])
    assert torch.max(torch.std(radar_frequencies, dim=0)).item() < 1e-6, "Radar data frequencies not identical!"
    radar_frequencies = radar_frequencies[0] # all frequencies asserted to be identical
    signal_real = torch.stack([torch.view_as_real(torch.from_numpy(radar_data.item().get('xr')[1:95, 1:95, :]).to(dtype=torch.cfloat, device=args['device'])).transpose(0, 1) for radar_data in radar_data_raw])
    signal_real_mean, signal_real_std = torch.view_as_real(torch.mean(torch.view_as_complex(signal_real), dim=0)), torch.std(signal_real, dim=0)
    print(f'Radar data layout: {signal_real.size(0)}x{tx_positions.size(0)}x{rx_positions.size(0)}x{radar_frequencies.size(-1)} <-> {torch.view_as_complex(signal_real).size()}')
    print(f'Radar signal value range: [{torch.min(signal_real):.2f}, {torch.max(signal_real):.2f}], mean: {torch.mean(torch.view_as_complex(signal_real)):.2f} stddev: {torch.std(torch.view_as_complex(signal_real)):.2f}')
    zeroed_antenna_mask = torch.all(torch.eq(signal_real_mean, torch.zeros_like(signal_real_mean)), dim=(-1, -2), keepdims=True)
    print(f'Zeroed antenna pairs: {torch.sum(zeroed_antenna_mask).item()}/{signal_real_mean.size(0)*signal_real_mean.size(1)}')

    # init differentiable radar
    radar = DiffRadarMaterial(
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        frequencies=radar_frequencies,
        tx_pattern=tx_pattern,
        rx_pattern=rx_pattern,
        mesh_vertices=vertices,
        mesh_indices=indices,
        material_type=args['material'],
        use_normalmap=args['use_normalmap'],
        use_reg_offset=args['use_reg_offset'],
        use_apc=args['use_apc'],
        material_storage=args['material_storage'])

    # ==============================================================
    # Inverse rendering

    if args['weights']:
        radar.load_state_dict(torch.load(args['weights']))
    if args['synthetic']:
        print(f'Generating synthetic radar data as optimization target...')
        # randomize target material parameters
        for param in radar.material.parameters():
            param.data[..., :] = torch.randn(param.size(-1)).cuda()
        # overwrite target signal with simulation
        with torch.no_grad():
            signal_real = torch.stack([radar.forward(args['seed']-signal_real.size(0)+i, args['n_hits']) for i in range(signal_real.size(0))])
            signal_real_mean, signal_real_std = torch.view_as_real(torch.mean(torch.view_as_complex(signal_real), dim=0)), torch.std(signal_real, dim=0)
        # re-randomize material parameters for subsequent optimization
        for param in radar.material.parameters():
            param.data = (torch.randn(param.size()).cuda() * 5).clamp_(-10, 10) # keep sigmoid behaved

    # setup tensorboard and json log
    tb_writer = SummaryWriter(comment=f'-{args["name"]}')
    json_log_data = {
        'name': args["name"],
        'date': str(datetime.datetime.now()),
        'config': dict(args),
        'l1': [],
        'l1_complex': [],
        'l2': [],
        'l2_reco': [],
        'gain_dB': [],
        'reg_offset': [],
    }
    with open(os.path.join(tb_writer.get_logdir(), f'log.json'), 'w', encoding='utf-8') as f:
        json.dump(json_log_data, f, ensure_ascii=False, indent=4)

    # visualize dataset, real radar data (target) and starting state
    with torch.no_grad():
        # visualize dataset (real)
        reco_target = radar.reco_from_signal(signal_real_mean, args['n_voxels'])
        rgb_images = [torchvision.transforms.functional.resize(torchvision.io.read_image(os.path.join(rgb_path, f)).to(torch.float32) / 255, 1024).cuda() for f in os.listdir(rgb_path) if '.JPG' in f]
        width, height = rgb_images[0].size(-1), rgb_images[0].size(-2)
        torchvision.utils.save_image([
            *rgb_images,
            radar.visualize_reco(reco_target, threshold_db=args['threshold_db'], width=width, height=height),
            radar.visualize_depth(width=width, height=height),
            radar.visualize_normals(width=width, height=height),
        ], os.path.join(tb_writer.get_logdir(), f'dataset.jpg'), nrow=2)
        torchvision.utils.save_image(radar.visualize_reco(reco_target, threshold_db=args['threshold_db']), os.path.join(tb_writer.get_logdir(), f'target_reco.jpg'))
        # visualize starting state (simulated)
        signal_start = radar.forward(args['seed'], args['n_hits'])
        reco_start = radar.reco_from_signal(signal_start, args['n_voxels'])
        torchvision.utils.save_image([
            radar.visualize_depth(),
            radar.visualize_normals(),
            radar.visualize_reco(reco_start, threshold_db=args['threshold_db']),
            radar.visualize_reco(reco_target, threshold_db=args['threshold_db']),
            radar.visualize_reco_diff(reco_start, reco_target, threshold_db=args['threshold_db']),
        ], os.path.join(tb_writer.get_logdir(), f'optim_reco_{0:04}.jpg'), nrow=5)
        torchvision.utils.save_image(radar.visualize_material(), os.path.join(tb_writer.get_logdir(), f'optim_mat_{0:04}.jpg'), nrow=2)

    optimizer = torch.optim.Adam(radar.parameters(), lr=args['lr'], betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args['n_epochs']//3), gamma=0.5)
    print(f'Model parameters: {sum(p.numel() for p in radar.parameters())}')

    # optimization loop
    for epoch in range(args['n_epochs']):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad()
        signal = radar.forward(args['seed'] + epoch, args['n_hits'])
        if args['use_zeroed']:
            signal = torch.where(zeroed_antenna_mask, torch.zeros_like(signal), signal)
        if args['loss'] == 'l1':
            loss = torch.nn.functional.l1_loss(signal, signal_real_mean)
        elif args['loss'] == 'l1_complex':
            loss = torch.nn.functional.l1_loss(torch.view_as_complex(signal), torch.view_as_complex(signal_real_mean))
        elif args['loss'] == 'l2':
            loss = torch.nn.functional.mse_loss(signal, signal_real_mean)
        elif args['loss'] == 'l2_reco':
            loss = 1000 * torch.nn.functional.mse_loss(radar.reco_from_signal(signal, args['n_voxels']) / torch.max(reco_target), reco_target / torch.max(reco_target))
        torch.cuda.synchronize()
        mid = time.perf_counter()
        loss.backward()
        optimizer.step()
        scheduler.step()
        for p in radar.parameters():
            p.data.clamp_(-10, 10) # clamp parms to sane values for sigmoid
        torch.cuda.synchronize()
        end = time.perf_counter()
        # Logging
        with torch.no_grad():
            # log to cmd line
            print(
                f'epoch {epoch+1:03}/{args["n_epochs"]:03}: '
                f'loss: {loss.item():0.3f}, '
                f'lr: {optimizer.param_groups[0]["lr"]:0.1e}, '
                f'max_grad: {max(torch.max(torch.abs(param.grad)).item() for param in radar.parameters() if param.grad is not None):0.1e}, '
                f'gain: {radar.gain_dB.item():0.2f}dB, '
                f'reg_off: ({0 if radar.reg_offset is None else torch.tanh(radar.reg_offset)[0].item():.1f}, {0 if radar.reg_offset is None else torch.tanh(radar.reg_offset)[1].item():.1f}, {0 if radar.reg_offset is None else torch.tanh(radar.reg_offset)[2].item():.1f}), '
                f'VRAM: {(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/1e9:.2f}GB, ',
                f'f/b/all: {mid - start:.1f}/{end - mid:.1f}/{end - start:.1f}s, ',
                f'ETA: {int((end-start)*(args["n_epochs"]-epoch-1)//(60*60))}h{int((end-start)*(args["n_epochs"]-epoch-1)//60)%60}m',
            flush=True)
            # compute metrics
            reco = radar.reco_from_signal(signal, args['n_voxels'])
            l1 = torch.nn.functional.l1_loss(signal.expand_as(signal_real), signal_real).item()
            l1_complex = torch.nn.functional.l1_loss(torch.view_as_complex(signal.expand_as(signal_real)), torch.view_as_complex(signal_real)).item()
            l2 = torch.nn.functional.mse_loss(signal, signal_real_mean).item()
            l2_reco = 1000 * torch.nn.functional.mse_loss(reco / torch.max(reco_target), reco_target / torch.max(reco_target)).item()
            # log to tensorboard
            tb_writer.add_scalar('Optim/loss', loss.item(), epoch)
            tb_writer.add_scalar('Optim/loss_L1', l1, epoch)
            tb_writer.add_scalar('Optim/loss_L1_complex', l1_complex, epoch)
            tb_writer.add_scalar('Optim/loss_L2', l2, epoch)
            tb_writer.add_scalar('Optim/loss_L2_reco', l2_reco, epoch)
            if epoch % max(1, args['n_epochs']//100) == 0:
                tb_writer.add_scalar('Optim/learning_rate', optimizer.param_groups[0]["lr"], epoch)
                tb_writer.add_scalar('Optim/VRAM_GB', (torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/1e9, epoch)
                tb_writer.add_scalar('Param/gain_dB', radar.gain_dB, epoch)
            # log to images
            if epoch < 20 or epoch % max(1, args['n_epochs']//30) == 0:
                torchvision.utils.save_image([
                        radar.visualize_depth(),
                        radar.visualize_normals(),
                        radar.visualize_reco(reco, threshold_db=args['threshold_db']),
                        radar.visualize_reco(reco_target, threshold_db=args['threshold_db']),
                        radar.visualize_reco_diff(reco, reco_target, threshold_db=args['threshold_db']),
                    ], os.path.join(tb_writer.get_logdir(), f'optim_reco_{epoch+1:04}.jpg'), nrow=5)
                torchvision.utils.save_image(radar.visualize_material(), os.path.join(tb_writer.get_logdir(), f'optim_mat_{epoch+1:04}.jpg'), nrow=2)
            # log to json
            json_log_data['l1'].append(l1)
            json_log_data['l1_complex'].append(l1_complex)
            json_log_data['l2'].append(l2)
            json_log_data['l2_reco'].append(l2_reco)
            json_log_data['gain_dB'].append(radar.gain_dB.item())
            json_log_data['reg_offset'].append((0, 0, 0) if radar.reg_offset is None else (torch.tanh(radar.reg_offset)[0].item(), torch.tanh(radar.reg_offset)[1].item(), torch.tanh(radar.reg_offset)[2].item()))
            with open(os.path.join(tb_writer.get_logdir(), f'log.json'), 'w', encoding='utf-8') as f:
                json.dump(json_log_data, f, ensure_ascii=False, indent=4)
            # save checkpoints at select epochs
            if epoch + 1 in [1, 10, 25, 50, 100, 250, 500, 1000]:
                torch.save(radar.state_dict(), os.path.join(tb_writer.get_logdir(), f'checkpoint_{epoch+1:04}.pt'))

    # save model to disk
    torch.save(radar.state_dict(), os.path.join(tb_writer.get_logdir(), f'radar.pt'))

    # create visualization of result
    with torch.no_grad():
        signal_result = radar.forward(args['seed'] + args['n_epochs'], args['n_hits'])
        reco_result = radar.reco_from_signal(signal_result, args['n_voxels'])
        torchvision.utils.save_image([
                radar.visualize_depth(),
                radar.visualize_normals(),
                radar.visualize_reco(reco_result, threshold_db=args['threshold_db']),
                radar.visualize_reco(reco_target, threshold_db=args['threshold_db']),
                radar.visualize_reco_diff(reco_result, reco_target, threshold_db=args['threshold_db']),
            ], os.path.join(tb_writer.get_logdir(), f'optim_reco_{args["n_epochs"]:04}.jpg'), nrow=5)
        torchvision.utils.save_image(radar.visualize_material(), os.path.join(tb_writer.get_logdir(), f'optim_mat_{args["n_epochs"]:04}.jpg'), nrow=2)
        torchvision.utils.save_image([
                radar.visualize_reco(reco_result, threshold_db=args['threshold_db']),
                radar.visualize_reco(reco_target, threshold_db=args['threshold_db']),
                radar.visualize_reco_diff(reco_result, reco_target, threshold_db=args['threshold_db']),
            ], os.path.join(tb_writer.get_logdir(), f'result_reco.jpg'), nrow=3)

    # create short clip of optimization process
    import imageio.v2 as iio
    mp4_reco = iio.get_writer(os.path.join(tb_writer.get_logdir(), 'optim_reco.mp4'), format='FFMPEG', mode='I', fps=4)
    mp4_mat = iio.get_writer(os.path.join(tb_writer.get_logdir(), 'optim_mat.mp4'), format='FFMPEG', mode='I', fps=4)
    for filename in sorted(os.listdir(tb_writer.get_logdir())):
        if 'optim_reco_' in filename and filename.endswith('.jpg'):
            mp4_reco.append_data(iio.imread(os.path.join(tb_writer.get_logdir(), filename)))
        if 'optim_mat_' in filename and filename.endswith('.jpg'):
            mp4_mat.append_data(iio.imread(os.path.join(tb_writer.get_logdir(), filename)))
    mp4_reco.close()
    mp4_mat.close()

    # clean up
    torch.cuda.synchronize()
    tb_writer.flush()
    tb_writer.close()

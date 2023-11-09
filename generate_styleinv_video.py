"""Generates a dataset of images using pretrained network pickle."""

import sys; sys.path.extend(['.', 'src'])
import os
import json
import random
import warnings

import click
import dnnlib
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

import pickle
import legacy
from training.logging import generate_videos, generate_videos_finetune, save_video_frames_as_mp4, save_video_frames_as_frames_parallel
import torchvision
torch.set_grad_enabled(False)

#----------------------------------------------------------------------------

def generate_latent_dataset(SI, G, outdir, num_videos, video_len, noise_mode='const'):
    basedir = os.path.split(outdir)[0]
    os.makedirs(basedir, exist_ok=True)
    G.eval()
    SI.eval()
    device = torch.device('cuda')
    ts = torch.arange(video_len, device=device).float().unsqueeze(0) # [1, video_len]
    ts /= (SI.mapping.cfg.sampling.max_num_frames - 1) # in StyleInV, clip [0, max-1] Ts are normalized to [0, 1]
    zc = torch.randn([num_videos, G.z_dim], device=device)
    with torch.no_grad():
        wc = G.mapping(zc, None)
    latents = []
    for i in tqdm(range(num_videos)):
        with torch.no_grad():
            x = G.synthesis(wc[i:i+1], noise_mode=noise_mode)
            codes = SI(x, wc[i:i+1, 0], None, ts)
        latents.append(codes.cpu())
    latents = torch.stack(latents)
    torch.save(latents, outdir)
    return

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', metavar='PATH')
@click.option('--networks_dir', help='Network pickles directory. Selects a checkpoint from it automatically based on the fvd2048_16f metric.', metavar='PATH')
@click.option('--truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--num_videos', type=int, help='Number of images to generate', default=50000, show_default=True)
@click.option('--batch_size', type=int, help='Batch size to use for generation', default=32, show_default=True)
@click.option('--moco_decomposition', type=bool, help='Should we do content/motion decomposition (available only for `--as_grids 1` generation)?', default=False, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--save_as_mp4', help='Should we save as independent frames or mp4?', type=bool, default=False, metavar='BOOL')
@click.option('--video_len', help='Number of frames to generate', type=int, default=16, metavar='INT')
@click.option('--fps', help='FPS for mp4 saving', type=int, default=25, metavar='INT')
@click.option('--as_grids', help='Save videos as grids', type=bool, default=False, metavar='BOOl')
@click.option('--time_offset', help='Additional time offset', default=0, type=int, metavar='INT')
@click.option('--dataset_path', help='Dataset path. In case we want to use the conditioning signal.', default="", type=str, metavar='PATH')
@click.option('--hydra_cfg_path', help='Config path', default="", type=str, metavar='PATH')
@click.option('--slowmo_coef', help='Increase this value if you want to produce slow-motion videos.', default=1, type=int, metavar='INT')
@click.option('--prepare_styleclip', help='Whether the scripts are used to save latent codes only', type=bool, default=False, metavar='BOOL')
@click.option('--finetune_pkl', help='Network for finetuned stylegan2', type=str, default='', metavar='PATH')
@click.option('--one_min_delta', help='One min delta', type=bool, default=False, metavar='BOOL')
def generate(
    ctx: click.Context,
    network_pkl: str,
    networks_dir: str,
    truncation_psi: float,
    noise_mode: str,
    num_videos: int,
    batch_size: int,
    moco_decomposition: bool,
    seed: int,
    outdir: str,
    save_as_mp4: bool,
    video_len: int,
    fps: int,
    as_grids: bool,
    time_offset: int,
    dataset_path: os.PathLike,
    hydra_cfg_path: os.PathLike,
    slowmo_coef: int,
    prepare_styleclip: bool,
    finetune_pkl: str,
    one_min_delta: bool
):
    if network_pkl is None:
        # output_regex = "^network-snapshot-\d{6}.pkl$"
        # ckpt_regex = re.compile("^network-snapshot-\d{6}.pkl$")
        # ckpts = sorted([f for f in os.listdir(networks_dir) if ckpt_regex.match(f)])
        # network_pkl = os.path.join(networks_dir, ckpts[-1])
        ckpt_select_metric = 'fvd2048_16f'
        metrics_file = os.path.join(networks_dir, f'metric-{ckpt_select_metric}.jsonl')
        with open(metrics_file, 'r') as f:
            snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
        best_snapshot = sorted(snapshot_metrics_vals, key=lambda m: m['results'][ckpt_select_metric])[0]
        network_pkl = os.path.join(networks_dir, best_snapshot['snapshot_pkl'])
        print(f'Using checkpoint: {network_pkl} with {ckpt_select_metric} of', best_snapshot['results'][ckpt_select_metric])
        # Selecting a checkpoint with the best score
    else:
        assert networks_dir is None, "Cant have both parameters: network_pkl and networks_dir"

    if moco_decomposition:
        assert as_grids, f"Content/motion decomposition is available only when we generate as grids."
        assert batch_size == num_videos, "Same motion is supported only for batch_size == num_videos"

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        ckpt = pickle.load(f)
        G = ckpt['G'].to(device).eval()
        si_key = 'SI_ema' if 'SI_ema' in ckpt.keys() else 'SI'
        SI = ckpt[si_key].to(device).eval()

    if finetune_pkl != '':
        concat_finetune = True
        with dnnlib.util.open_url(finetune_pkl) as f:
            ckpt = pickle.load(f)
            G_tuned = ckpt['G_ema'].to(device).eval()
    else:
        concat_finetune = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if prepare_styleclip:
        generate_latent_dataset(SI, G, outdir, num_videos, video_len, noise_mode=noise_mode)
        return 

    #model_desc = network_pkl.split('/')[-3]
    #outdir = os.path.join(outdir, model_desc)
    os.makedirs(outdir, exist_ok=True)

    all_zc = torch.randn([num_videos, G.z_dim], device=device) # [curr_batch_size, z_dim]
    with torch.no_grad():
        all_wc = G.mapping(all_zc, None)
    all_c = torch.zeros([num_videos, G.c_dim], device=device) # [num_videos, c_dim]

    if one_min_delta:
        ts = torch.arange(0, 128, 16, device=device).float()
        delta = 128 + 30 * 60
        ts_delta = torch.arange(delta, delta + 128, 16, device=device).float()
        ts = torch.cat([ts, ts_delta])
        ts = ts.unsqueeze(0).repeat(batch_size, 1)
    else:
        ts = time_offset + torch.arange(video_len, device=device).float().unsqueeze(0).repeat(batch_size, 1) / slowmo_coef # [batch_size, video_len]
    
    ts /= (SI.mapping.cfg.sampling.max_num_frames - 1) # in StyleInV, clip [0, max-1] Ts are normalized to [0, 1]
    ts = ts.unsqueeze(-1) # (batch_size, video_len, 1)

    if moco_decomposition:
        num_rows = num_cols = int(np.sqrt(num_videos))
        motion_z = SI.mapping.motion_encoder(c=all_c[:num_rows], t=ts[:num_rows])['motion_z'] # [1, *motion_dims]
        motion_z = motion_z.repeat_interleave(num_cols, dim=0) # [batch_size, *motion_dims]

        all_wc = all_wc[:num_cols].repeat(num_rows, 1) # [num_videos, z_dim]
        all_c = all_c[:num_cols].repeat(num_rows, 1) # [num_videos, z_dim]
    else:
        motion_z = None

    # Generate images.
    for batch_idx in tqdm(range((num_videos + batch_size - 1) // batch_size), desc='Generating videos'):
        curr_batch_size = batch_size if batch_size * (batch_idx + 1) <= num_videos else num_videos % batch_size
        wc = all_wc[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] # [curr_batch_size, z_dim]
        c = all_c[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] # [curr_batch_size, c_dim]

        if not concat_finetune:
            videos = generate_videos(
                SI, G, wc, c, ts[:curr_batch_size], motion_z=motion_z, noise_mode=noise_mode,
                truncation_psi=truncation_psi, as_grids=as_grids, batch_size_num_frames=128)
        else:
            videos = generate_videos_finetune(SI, G, G_tuned, wc, c, ts[:curr_batch_size], motion_z=motion_z, noise_mode=noise_mode,
                truncation_psi=truncation_psi, as_grids=as_grids, batch_size_num_frames=128)

        if as_grids:
            videos = [videos]

        for video_idx, video in enumerate(videos):
            if save_as_mp4:
                save_path = os.path.join(outdir, f'{batch_idx * batch_size + video_idx:06d}.mp4')
                video = (video * 255).permute(0, 2, 3, 1).to(torch.uint8) # (t, h, w, c)
                torchvision.io.write_video(save_path, video, fps=fps, video_codec='h264', options={'crf': '10'})
            else:
                save_dir = os.path.join(outdir, f'{batch_idx * batch_size + video_idx:06d}')
                video = (video * 255).permute(0, 2, 3, 1).to(torch.uint8).numpy() # [video_len, h, w, c]
                save_video_frames_as_frames_parallel(video, save_dir, time_offset=time_offset, num_processes=8)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

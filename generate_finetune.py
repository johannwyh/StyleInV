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
from PIL import Image

import pickle
import legacy
from training.logging import generate_videos, save_video_frames_as_mp4, save_video_frames_as_frames_parallel
import torchvision
torch.set_grad_enabled(False)

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', metavar='PATH')
@click.option('--network_raw_pkl', help='Network before finetune pickle filename.', metavar='PATH')
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
@click.option('--latent_file', help='Stored StyleInV generated videos latents', default='', type=str, metavar='PATH')
@click.option('--save_films', help='Stored every eighth frame to form a film', type=bool, default=False, metavar='BOOL')
def generate(
    ctx: click.Context,
    network_pkl: str,
    network_raw_pkl: str,
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
    latent_file: str,
    save_films: bool
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with open(network_pkl, 'rb') as f:
        ckpt = pickle.load(f)
        G = ckpt['G_ema'].to(device).eval()
    
    print('Loading raw networks from "%s"...' % network_raw_pkl)
    with open(network_raw_pkl, 'rb') as f:
        ckpt = pickle.load(f)
        G_raw = ckpt['G_ema'].to(device).eval()

    latent_bank = torch.load(latent_file)
    num_ws = G.synthesis.num_ws
    os.makedirs(outdir, exist_ok=True)
    for i in tqdm(range(latent_bank.shape[0])):
        ws = latent_bank[i].to(device)
        ws = ws.unsqueeze(1).repeat([1, num_ws, 1])
        video = G.synthesis(ws, noise_mode=noise_mode)
        video = (video * 0.5 + 0.5).clamp(0, 1).cpu()
        video_raw = G_raw.synthesis(ws, noise_mode=noise_mode)
        video_raw = (video_raw * 0.5 + 0.5).clamp(0, 1).cpu()

        if save_films:
            tuned_frames = video[:128:8] # (16, 3, 256, 256)
            raw_frames = video_raw[:128:8] # (16, 3, 256, 256)
            tfs_, rfs_ = [], []
            for j in range(tuned_frames.shape[0]):
                tframe = tuned_frames[j]
                rframe = raw_frames[j]
                tf_ = torch.ones(3, 262, 262 + 5)
                tf_[:, 3:-3, 3:-8] = tframe
                tf_[:, :, -5:] = 0
                tfs_.append(tf_)
                rf_ = torch.ones(3, 262, 262 + 5)
                rf_[:, 3:-3, 3:-8] = rframe
                rf_[:, :, -5:] = 0
                rfs_.append(rf_)
            tfs = torch.cat(tfs_, dim=2)
            rfs = torch.cat(rfs_, dim=2)
            film = torch.cat([rfs, torch.zeros(3, 5, tfs.shape[-1]), tfs], dim=1)
            film = (film * 255).permute(1,2,0).to(torch.uint8).numpy()
        video = torch.cat([video_raw, video], dim=-1)

        if save_as_mp4:
            save_path = os.path.join(outdir, f'{i:06d}.mp4')
            video = (video * 255).permute(0, 2, 3, 1).to(torch.uint8) # (t, h, w, c)
            torchvision.io.write_video(save_path, video, fps=fps, video_codec='h264', options={'crf': '10'})
            if save_films:
                save_film_path = os.path.join(outdir, f'{i:06d}.jpg')
                Image.fromarray(film).save(save_film_path, q=95)
        else:
            save_dir = os.path.join(outdir, f'{i:06d}')
            video = (video * 255).permute(0, 2, 3, 1).to(torch.uint8).numpy() # [video_len, h, w, c]
            save_video_frames_as_frames_parallel(video, save_dir, time_offset=time_offset, num_processes=8)

if __name__ == "__main__":
    generate() # pylint: disable=no-value-for-parameter
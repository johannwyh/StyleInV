import sys; sys.path.extend(['.', 'src'])
import os
import json
import random
import warnings

import PIL
from PIL import Image
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

from face_util.ffhq_cropper import FaceCropper

def pil_to_torch(img):
    # [H, W, C] -> [C, H, W]
    # [0, 255] -> [-1, 1]
    img_np = np.array(img).astype(np.float32).transpose(2, 0, 1)
    img_torch = torch.from_numpy(img_np) / 127.5 - 1
    return img_torch

def torch_to_pil(img):
    img_np = ((img.cpu() * 0.5 + 0.5).clamp(0, 1) * 255).numpy()
    img_np = img_np.astype(np.uint8).transpose(1, 2, 0)
    return PIL.Image.fromarray(img_np, 'RGB')

def inv(img, pSp, G):
    w_inv = pSp(img)
    out = G.synthesis(w_inv)
    return out, w_inv

@click.command()
@click.pass_context
@click.option('--styleinv_pkl', help='StyleInV Network pickle filename', metavar='PATH')
@click.option('--psp_pkl', help='pSp Network pickle filename.', metavar='PATH')
@click.option('--noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--num_videos', type=int, help='Number of images to generate', default=50000, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--inputdir', help='Where the input samples are saved', type=str, required=True, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--save_as_mp4', help='Should we save as independent frames or mp4?', type=bool, default=False, metavar='BOOL')
@click.option('--video_len', help='Number of frames to generate', type=int, default=16, metavar='INT')
@click.option('--fps', help='FPS for mp4 saving', type=int, default=25, metavar='INT')
@click.option('--time_offset', help='Additional time offset', default=0, type=int, metavar='INT')
@click.option('--finetune_pkl', help='Network for finetuned stylegan2', type=str, default='', metavar='PATH')
@torch.no_grad()
def generate(
    ctx: click.Context,
    styleinv_pkl: str,
    psp_pkl: str,
    noise_mode: str,
    num_videos: int,
    seed: int,
    inputdir: str,
    outdir: str,
    save_as_mp4: bool,
    video_len: int,
    fps: int,
    time_offset: int,
    finetune_pkl: str,
):
    cropper = FaceCropper()
    print('Loading StyleInV network from "%s"...' % styleinv_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(styleinv_pkl) as f:
        ckpt = pickle.load(f)
        G = ckpt['G'].to(device).eval()
        si_key = 'SI_ema' if 'SI_ema' in ckpt.keys() else 'SI'
        SI = ckpt[si_key].to(device).eval()

    print('Loading pSp network from "%s"...' % psp_pkl)
    with dnnlib.util.open_url(psp_pkl) as f:
        ckpt = pickle.load(f)
        pSp = ckpt['pSp'].to(device).eval()
    
    if finetune_pkl != '':
        concat_finetune = True
        finetune_name = os.path.basename(finetune_pkl).split('_')[0]
        print('Loading style transfer network from "%s"...' % finetune_pkl)
        with dnnlib.util.open_url(finetune_pkl) as f:
            ckpt = pickle.load(f)
            G_tuned = ckpt['G_ema'].to(device).eval()

    first_frame_files = sorted(os.listdir(inputdir))
    # debug
    # first_frame_files = first_frame_files[:3]

    ts = time_offset + torch.arange(video_len, device=device).float().unsqueeze(0) # [batch_size, video_len]
    ts /= (SI.mapping.cfg.sampling.max_num_frames - 1) # in StyleInV, clip [0, max-1] Ts are normalized to [0, 1]
    ts = ts.unsqueeze(-1) # (batch_size, video_len, 1)

    os.makedirs(outdir, exist_ok=True)

    for fn in tqdm(first_frame_files):
        if fn.endswith('avif'):
            continue
        full_fn = os.path.join(inputdir, fn)

        save_fn = os.path.splitext(fn)[0]
        if concat_finetune:
            save_fn += f'-with_{finetune_name}'
        out_fn = os.path.join(outdir, save_fn + '.mp4')

        # 256 ffhq cropped
        img_256_pil = cropper.crop_ffhq(full_fn, output_size=256, transform_size=1024)[0]
        img_256 = pil_to_torch(img_256_pil).unsqueeze(0).cuda()

        # inv, w0
        inv_256, wc0 = inv(img_256, pSp, G)

        # styleinv
        num_ws = wc0.shape[1]
        wc0 = wc0[:, 0, :]
        motion_z = SI.mapping.generate_motion_sequence(ts)
        styles = SI(inv_256, wc0, motion_z, ts, run_parallel=True, return_temporal_style=False)

        # synthesis
        styles = styles.unsqueeze(1).repeat([1, num_ws, 1])
        frames = G.synthesis(styles, noise_mode=noise_mode).cpu() # (T, C, H, W)
        if concat_finetune:
            frames_trans = G_tuned.synthesis(styles, noise_mode=noise_mode).cpu() # (T, C, H, W)
            frames = torch.cat([frames, frames_trans], dim=3) # (T, C, H, W * 2)

        # save
        img_256 = img_256.cpu().repeat([video_len, 1, 1, 1])
        inv_256 = inv_256.cpu().repeat([video_len, 1, 1, 1]) 
        frames = torch.cat([img_256, inv_256, frames], dim=3) # (T, C, H, W*k)
        frames = ((frames * 0.5 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).to(torch.uint8)
        torchvision.io.write_video(out_fn, frames, fps=fps, video_codec='h264', options={'crf': '10'})

        latents_dict = {
            'input_fn': fn,
            'wc0': wc0.cpu(),
            'styles': styles[:, 0, :].cpu()
        }
        torch.save(latents_dict, os.path.join(outdir, save_fn + '.pt'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
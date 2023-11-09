# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train StyleInV"."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import yaml
from omegaconf import OmegaConf
from omegaconf import DictConfig
from training import training_loop_styleinv, training_loop_styleinv_test
from metrics_styleinv import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from hydra import initialize, compose
from copy import deepcopy
#----------------------------------------------------------------------------

class UserError(Exception):
    pass

augpipe_specs = {
    'blit':      dict(xflip=1, rotate90=1, xint=1),
    'geom':      dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color':     dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter':    dict(imgfilter=1),
    'noise':     dict(noise=1),
    'cutout':    dict(cutout=1),
    'bg':        dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc':       dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'bgcf':      dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
    'bgcfn':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    'easy':      dict(xflip=1, xint=1, scale=1, rotate=0.5, rotate_max=0.1, xfrac=1, noise=0.1, cutout=1, cutout_size=0.25),
    'bgc_norgb': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, cutout=1),
}

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    seed       = None, # Random seed: <int>, default = 0

    # Base config.
    cfile      = None, # Dataset related config file, 
    overrides  = None, # Overrides arguments in cfile, splitted with ','
    cname      = None, # Base config: '4c256' (default), '8c256'
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    resume_whole_state = None, # Whether to resume all modules
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # Load pre-written config file
    config_dir = os.path.dirname(cfile)
    config_file = os.path.basename(cfile)
    initialize(config_path=config_dir, job_name='styleinv')
    if overrides is None:
        overrides = []
    else:
        overrides = overrides.split(',')
    cfg = compose(config_file, overrides=overrides)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    metrics = cfg.metrics
    assert len(metrics) >= 0
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # ---------
    # Visualize
    # ---------
    args.visualize_args = cfg.visualize_args

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    data = cfg.dataset.data
    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset_styleinv.StyleInVDataset', path=data, resolution=cfg.dataset.size, nframes=cfg.dataset.nframes, stride=cfg.dataset.stride, xflip=cfg.dataset.mirror, flex_sampling=cfg.dataset.flex_sampling, random_offset=True)
    args.training_set_kwargs.sampling_cfg = cfg.real_sampling
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)

    dataset_stata = dnnlib.util.EasyDict()
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        dataset_stata.resolution = training_set.img_resolution # be explicit about resolution
        dataset_stata.total_videos = training_set.n_videos
        dataset_stata.total_frames = training_set.total_frames
        dataset_stata.xflip = training_set.xflip
        dataset_stata.sampling_type = training_set.sampling_cfg.type
        desc = f'{training_set.name}_{dataset_stata.sampling_type}'
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cname is None:
        cname = '4c256'
    assert isinstance(cname, str)
    desc += f'-{cname}'

    cfg_specs = {
        '8c256':     dict(ref_gpus=8, kimg=25000, mb=64, mbstd=8, fmaps=0.5, ema=20, ramp=None),
        '6c256':     dict(ref_gpus=6, kimg=25000, mb=48, mbstd=8, fmaps=0.5, ema=20, ramp=None),
        '4c256':     dict(ref_gpus=4, kimg=25000, mb=32, mbstd=8, fmaps=0.5, ema=20, ramp=None),
        '2c256':     dict(ref_gpus=2, kimg=12000, mb=16, mbstd=8, fmaps=0.5, ema=20, ramp=None),
        '8c1024':    dict(ref_gpus=8, kimg=25000, mb=32, mbstd=4, fmaps=1, ema=10, ramp=None),
        '4c1024':    dict(ref_gpus=4, kimg=25000, mb=16, mbstd=4, fmaps=1, ema=10, ramp=None),
        '8c256_batch1': dict(ref_gpus=8, kimg=25000, mb=8, mbstd=1, fmaps=0.5, ema=20, ramp=None),
        '8c256_batch2': dict(ref_gpus=8, kimg=25000, mb=16, mbstd=2, fmaps=0.5, ema=20, ramp=None),
        '8c256_batch4': dict(ref_gpus=8, kimg=25000, mb=32, mbstd=4, fmaps=0.5, ema=20, ramp=None),
    }

    assert cname in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cname])

    # checkpoint stores psp and stylegan2 pkl, saved by psp training code
    args.checkpoint = cfg.checkpoint

    # models
    styleinv_fps = cfg.dataset.fps / cfg.dataset.stride if cfg.dataset.stride > 1 else cfg.dataset.fps
    args.SI_kwargs = dnnlib.EasyDict(
        class_name='training.networks_styleinv.StyleInV', 
        checkpoint=args.checkpoint, 
        mapping_opts=cfg.styleinv.mapping_opts, 
        encoder_opts=cfg.styleinv.encoder_opts, 
        input_residual=cfg.styleinv.input_residual, 
        clip_len=cfg.dataset.nframes, 
        fps=styleinv_fps, 
        mutual_recon=cfg.mutual_info, 
        skip_encoder=cfg.styleinv.skip_encoder
    )

    assert cfg.discriminator.type in ['digan', 'stylegan-v', 'ffc']
    if cfg.discriminator.type == 'digan':
        channels = cfg.real_sampling.num_frames_per_video * 4 - 1
        args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', c_dim=0, img_resolution=cfg.dataset.size, img_channels=channels, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    elif cfg.discriminator.type == 'stylegan-v':
        args.D_kwargs = dnnlib.EasyDict(class_name=f'training.networks_sgv.Discriminator', c_dim=0, img_resolution=cfg.dataset.size, img_channels=3, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict(), cfg=cfg.discriminator.cfg)
        args.D_kwargs.mapping_kwargs.num_layers = 2
    elif cfg.discriminator.type == 'ffc':
        args.D_kwargs = dnnlib.EasyDict(class_name=f'training.networks_styleinv.DiscriminatorFFC', c_dim=0, img_resolution=cfg.dataset.size, img_channels=3, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict(), cfg=cfg.discriminator.cfg)
        args.D_kwargs.mapping_kwargs.num_layers = 2
        
    args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.D_kwargs.channel_max = 512
    args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
        

    args.SI_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=cfg.lrSI, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=cfg.lrD, betas=[0,0.99], eps=1e-8)

    # Here we set noise_mode default to const in case some versions of model requrie random noise
    # For no-noise mode, this is hard-encoded into the image generator itself and won't be affected by this setting
    noise_mode = cfg.get('noise_mode', 'const') 
    
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_styleinv.StyleInVLoss', 
        mutual_recon=cfg.mutual_info, 
        first_frame_recon=cfg.loss_kwargs.first_frame_recon, 
        first_frame_in_D=cfg.loss_kwargs.first_frame_in_D, 
        first_frame_use_gen=cfg.loss_kwargs.first_frame_use_gen, 
        run_g_parallel=True, 
        r1_gamma=cfg.loss_kwargs.lambdas.r1_gamma, 
        lambda_adv=cfg.loss_kwargs.lambdas.lambda_adv, 
        lambda_mutual=cfg.loss_kwargs.lambdas.lambda_mutual, 
        lambda_recon=cfg.loss_kwargs.lambdas.lambda_recon, 
        noise_mode=noise_mode
    )
    lambda_w_reg = cfg.loss_kwargs.lambdas.get('lambda_w_reg', 0)
    args.loss_kwargs.lambda_w_reg = lambda_w_reg
    args.loss_kwargs.g_sampling_cfg = cfg.g_sampling
    args.loss_kwargs.real_sampling_cfg = cfg.real_sampling
    args.loss_kwargs.D_type = cfg.discriminator.type

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp
    
    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ----------------------------------
    # Augmentation
    # ----------------------------------

    assert cfg.aug in ['noaug', 'ada'], f'Augmentation method {cfg.aug} not supported'
    desc += f'-aug_{cfg.aug}'
    if cfg.aug == 'ada':
        # We focused on using ADA for now, so args.augment_p = 0
        args.ada_target = 0.6
        if cfg.target is not None:
            assert isinstance(cfg.target, float), 'In ada mode, target must be a float number'
            assert 0 <= cfg.target <= 1, f'target value {cfg.target} is out of range [0, 1]'
            desc += f'-target{cfg.target}'
            args.ada_target = cfg.target
        assert cfg.augpipe in augpipe_specs.keys(), f'augpipe {cfg.augpipe} not supported'
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment_styleinv.AugmentPipe', **augpipe_specs[cfg.augpipe])
    else:
        args.augment_kwargs = None

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url
        if resume_whole_state is None:
            resume_whole_state = True
        args.resume_whole_state = resume_whole_state

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.D_kwargs.num_fp16_res = 0
        args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args, dataset_stata

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop_styleinv.training_loop(rank=rank, **args)
    #training_loop_styleinv_test.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Base config.
@click.option('--cfile', help='Config file used to store dataset related configs', required=True, type=str, metavar='DIR')
@click.option('--overrides', help='Arguments that are overrided from the raw config yaml', type=str, metavar='STR')
@click.option('--cname', help='Base config [default: 4c256]', type=click.Choice(['8c256', '4c256', '8c256_batch1', '8c256_batch2', '8c256_batch4', '2c256', '6c256']))
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--resume_whole_state', help='If set, resume all modules', type=bool, metavar='BOOL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args, dataset_stata = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    args_omega = OmegaConf.create(dnnlib.util.convert_easy_dict_to_dict(args))
    print(OmegaConf.to_yaml(args_omega, resolve=True))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of videos:   {dataset_stata.total_videos}')
    print(f'Number of frames:   {dataset_stata.total_frames}')
    print(f'Image resolution:   {dataset_stata.resolution}')
    print(f'Dataset x-flips:    {dataset_stata.xflip}')
    print(f'Dataset Sampling:   {dataset_stata.sampling_type}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.yaml'), 'w') as f:
        OmegaConf.save(config=args_omega, f=f, resolve=True)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

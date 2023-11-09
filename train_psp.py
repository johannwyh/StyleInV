import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop_psp
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from training.dataset import PSP_Transform_Dict
from auxiliary.config import psp_model_paths

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    data       = None, # Training dataset (required): <path>
    video_balance = None, # Video Balance for dataset

    # Pretrained Models.
    sg2_pkl    = None, # Pretrained weights for StyleGAN2 pkl: <path>
    noise_mode = None, # StyleGAN2 noise mode

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>
    optim      = None, # Specify optimizer type: <str>, default = 'Ranger' (default), 'Adam'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>

    # Performance options (not included in desc).
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
    suffix     = None, # Outdir desc suffix
):
    args = dnnlib.EasyDict()

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

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    if video_balance is None:
        video_balance = False
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.PSPDataset', source_root=data, target_root=data, label_nc=0, transform_type='train', video_balance=video_balance)
    args.test_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.PSPDataset', source_root=data, target_root=data, label_nc=0, transform_type='test', video_balance=video_balance)

    args.train_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    args.test_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2, shuffle=False, drop_last=True)

    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        test_set = dnnlib.util.construct_class_by_name(**args.test_set_kwargs) # subclass of training.dataset.Dataset
        dataset_size = len(training_set.target_paths)
        desc = training_set.name
        del training_set # conserve memory
        del test_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')


    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'psp_paper'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'psp_paper':     dict(ref_gpus=8, kimg=4000,  mb=64, mbstd=8, lrate=0.0001, id_lambda=0.1, l2_lambda=1, lpips_lambda=0.8), # Original setting for training a PSP encoder with DFUF-256 dataset.
        'psp_auto':      dict(ref_gpus=-1, kimg=4000,  mb=-1, mbstd=8, lrate=0.0001, id_lambda=0.1, l2_lambda=1, lpips_lambda=0.8), # Original setting for training a PSP encoder with DFUF-256 dataset.
        'psp_sky_auto':  dict(ref_gpus=-1, kimg=4000,  mb=-1, mbstd=8, lrate=0.0001, id_lambda=0, l2_lambda=1, lpips_lambda=0.8), # Setting for sky-256 inversion
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if 'auto' in cfg:
        spec.ref_gpus = args.num_gpus
        spec.mb = args.num_gpus * spec.mbstd

    assert sg2_pkl is not None
    args.G_pkl = sg2_pkl

    if noise_mode is None:
        noise_mode = 'const'
    args.noise_mode = noise_mode
    
    irse50 = psp_model_paths['ir_se50']
    pSp_opts = dnnlib.EasyDict(output_size=256, encoder_type="BackboneEncoderUsingLastLayerIntoW", irse50=irse50, start_from_latent_avg=True, learn_in_w=True, input_nc=3)
    args.PSP_kwargs = dnnlib.EasyDict(class_name='training.networks_psp.pSp', opts=pSp_opts)

    lpips_kwargs = dnnlib.EasyDict(class_name='auxiliary.lpips.lpips.LPIPS', net_type='alex')
    id_kwargs = dnnlib.EasyDict(class_name='auxiliary.id_loss.IDLoss')
    args.auxiliary_kwargs = dnnlib.EasyDict(lpips_kwargs=lpips_kwargs, id_kwargs=id_kwargs)

    if optim is None:
        optim = 'Ranger'
    assert optim is not None and optim in ['Ranger', 'Adam']
    if optim == 'Adam':
        args.PSP_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate)
    else:
        args.PSP_opt_kwargs = dnnlib.EasyDict(class_name='auxiliary.ranger.Ranger', lr=spec.lrate)

    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_psp.pSpLoss', lpips_lambda=spec.lpips_lambda, l2_lambda=spec.l2_lambda, id_lambda=spec.id_lambda)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus
    
    #  Override the batch size of dataloaders here
    args.train_loader_kwargs.batch_size = args.test_loader_kwargs.batch_size = args.batch_gpu

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

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
        args.train_loader_kwargs.num_workers = workers
        args.test_loader_kwargs.num_workers = workers

    if suffix is None:
        suffix = ''
    desc += f'_{suffix}'
    return desc, args, dataset_size

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
    training_loop_psp.training_loop(rank=rank, **args)

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

# Dataset.
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--video-balance', help='Video Balance for Dataset', is_flag=True)

# Pretrained Models
@click.option('--sg2-pkl', help='Pretrained StyleGAN2 pkl', metavar='PKL')
@click.option('--noise-mode', help='StyleGAN2 noise mode', metavar='STR')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['psp_paper', 'psp_auto', 'psp_sky_auto']))
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')
@click.option('--optim', help='Specify optimizer', type=click.Choice(['Adam', 'Ranger']))

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')

# Performance options.
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')
@click.option('--suffix', help='Outdir desc suffix', type=str, metavar='DIR')

def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN Inversion model PSP
    Supported Encoder: BackboneEncoderUsingLastLayerIntoW
    To be Supported: BackboneEncoderUsingLastLayerIntoWPlus, GradualStyleEncoder
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args, dataset_size = setup_training_loop_kwargs(**config_kwargs)
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
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.source_root}')
    print(f'Training duration:  {args.total_kimg} kimg, {int(args.total_kimg * 1000 / args.batch_size)} iters')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {dataset_size}')
    print(f'Image resolution:   {256}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
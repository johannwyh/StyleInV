# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import random
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import torchvision

import legacy
from metrics_styleinv import metric_main
from einops import rearrange
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw, gh = 5, 5

    # No labels => show random subset of training samples.
    training_set.return_one = True
    all_indices = list(range(training_set.total_frames))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images = [training_set[i].copy() for i in grid_indices]
    training_set.return_one = False
    return (gw, gh), np.stack(images)

def convert_batch_videos_to_grid(batch_videos, grid_size): # (b,c,t,h,w) -> (t, H, W, c)
    gw, gh = grid_size
    _N, C, T, H, W = batch_videos.shape
    videos = batch_videos.reshape(gh, gw, C, T, H, W)
    videos = videos.permute(3, 0, 4, 1, 5, 2)
    videos = videos.reshape(T, gh*H, gw*W, C)
    return videos

def setup_snapshot_video_grid(training_set_kwargs, grid_size=(5,5), random_seed=0):
    gw, gh = grid_size
    rnd = np.random.RandomState(random_seed)
    svdata_kwargs = copy.deepcopy(training_set_kwargs)
    svdata_kwargs.return_vid = True
    svdata = dnnlib.util.construct_class_by_name(**svdata_kwargs)
    all_indices = list(range(svdata.n_videos))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    videos = torch.cat([torch.from_numpy(svdata[i]).unsqueeze(0) for i in grid_indices]) # (b, c, t, h, w)
    del svdata
    return convert_batch_videos_to_grid(videos, grid_size)
#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def generate_visualize_videos(lstm, G_synthesis, grid_batch=1, grid_wc=None, nframes=128, grid_num_ws=14, noise_mode='none'):
    grid_videos, grid_image = [], []
    for grid_idx in range(grid_batch):
        grid_styles, _, _ = lstm(grid_wc[grid_idx:grid_idx+1], nframes) # ((b t) c) = (t c)
        grid_styles_b = grid_styles.unsqueeze(1).repeat([1, grid_num_ws, 1])
        grid_vid = G_synthesis(grid_styles_b, noise_mode=noise_mode).cpu() # (t, c, h, w)
        grid_image.append(grid_vid[0:1])
        grid_videos.append(grid_vid.unsqueeze(0))
    
    grid_images = torch.cat(grid_image).numpy() # (b, c, h, w)
    grid_videos = torch.cat(grid_videos) # (b, t, c, h, w)
    assert grid_videos.ndim == 5
    grid_videos = grid_videos.permute(0, 2, 1, 3, 4) # (b, c, t, h, w)
    # grid_videos: 
    return grid_videos, grid_images

def training_loop(
    run_dir                 = '.',      # Output directory.
    visualize_args          = {},       # Visualize args: viz_len, fps
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    checkpoint              = '',       # PKL of pretrained pSp and StyleGAN2
    lstm_kwargs             = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    SI_kwargs               = {},
    lstm_opt_kwargs         = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_whole_state      = False,    # Should we resume the whole state or only the G/D/G_ema checkpoints?
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    this_random_seed = random_seed * num_gpus + rank
    random.seed(this_random_seed)
    np.random.seed(this_random_seed)
    torch.manual_seed(this_random_seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num videos: ', training_set.n_videos)
        print('Num frames: ', training_set.total_frames)
        print('Image shape:', training_set.img_resolution)
        print('Sampling: ', training_set.sampling_cfg.type)
        print()
    nframes = training_set.nframes

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    with open(checkpoint, 'rb') as f:
        G = pickle.load(f)['G'].eval().requires_grad_(False).to(device)
    loss_kwargs.num_ws = G.synthesis.num_ws
    lstm = dnnlib.util.construct_class_by_name(**lstm_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    lstm_ema = copy.deepcopy(lstm).eval()

    # Resume from existing pickle.
    if resume_pkl is not None:
        print(f'Resuming from "{resume_pkl}"')
        with open(resume_pkl, 'rb') as f:
            resume_data = pickle.load(f)
        for name, module in [('G', G), ('D', D), ('lstm', lstm), ('lstm_ema', lstm_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    content_dim = 512
    noise_mode = loss_kwargs.noise_mode
    if rank == 0:
        # prepare inputs
        zc = torch.randn([batch_gpu, content_dim], device=device)
        g_frames = SI_kwargs.mapping_opts.sampling.num_frames_per_video
        d_frames = loss_kwargs.real_sampling_cfg.num_frames_per_video
        Ts = torch.linspace(0, 1, steps=g_frames).view(g_frames, 1).unsqueeze(0).to(device)
        Ts = Ts.repeat([batch_gpu, 1, 1])
        
        # Generate initial frame
        wc = misc.print_module_summary(G.mapping, [zc, None])
        num_ws = wc.shape[1]
        img0 = misc.print_module_summary(G.synthesis, [wc])
        wc = wc[:, 0, :]
        
        # StyleInV output latents
        styles, _, _ = misc.print_module_summary(lstm, [wc, nframes]) # (b t) d
        styles = styles[:g_frames*batch_gpu] 
        styles_broadcast = styles.unsqueeze(1).repeat([1, num_ws, 1])
        
        # Map latents to images
        imgs = misc.print_module_summary(G.synthesis, [styles_broadcast])
        img_list = list(imgs.split(batch_gpu, dim=0))

        # process adversarial input
        if g_frames < d_frames:
            img_list = [img0] + img_list
            Ts = torch.cat([torch.zeros(batch_gpu, 1, 1).float().to(device), Ts], dim=1)
        elif g_frames > d_frames:
            img_list = img_list[1:]
            Ts = Ts[:, 1:]

        # Quintuplet sparse training discriminator
        if loss_kwargs.D_type == 'digan':
            img_D = torch.cat(img_list, dim=1)
            H, W = img_D.shape[2:]
            Ts = Ts.unsqueeze(-1)
            dTs = (Ts[:, 1:] - Ts[:, :-1]).repeat(1, 1, H, W)
            img_D = torch.cat([img_D, dTs], dim=1)
            misc.print_module_summary(D, [img_D, None])
        elif loss_kwargs.D_type == 'stylegan-v':
            Ts = Ts.squeeze(-1)
            img_D = torch.cat(img_list, dim=1)
            img_D = rearrange(img_D, 'b (t c) h w -> (b t) c h w', c=3)
            misc.print_module_summary(D, [img_D, Ts])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')

    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))

        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')
        else:
            ada_stats = None

        if resume_whole_state:
            misc.copy_params_and_buffers(resume_data['augment_pipe'], augment_pipe, require_all=False)
    else:
        augment_pipe = None
        ada_stats = None

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('lstm', lstm), (None, lstm_ema), ('D', D), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, nframes=nframes, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    if rank == 0:
        print(f"StyleInV loss: input_skip_to_D = {loss.input_skip_to_D}")
        
    phases = []
    for name, module, opt_kwargs, reg_interval in [('lstm', lstm, lstm_opt_kwargs, 0), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            if reg_interval == 0:
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            else:        
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images and videos ...')
        grid_size, images = setup_snapshot_image_grid(training_set=training_set, random_seed=this_random_seed)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        real_videos = setup_snapshot_video_grid(training_set_kwargs=training_set_kwargs, grid_size=grid_size, random_seed=this_random_seed)
        torchvision.io.write_video(os.path.join(run_dir, 'reals.mp4'), real_videos, fps=visualize_args.fps, video_codec='h264', options={'crf': '10'})

        grid_batch = grid_size[0] * grid_size[1]
        grid_zc = torch.randn([grid_batch, content_dim], device=device)
        grid_wc = G.mapping(grid_zc, None)
        grid_num_ws = grid_wc.shape[1]
        grid_wc = grid_wc[:, 0, :]
        grid_args = dnnlib.EasyDict(grid_batch=grid_batch, grid_wc=grid_wc, nframes=nframes, grid_num_ws=grid_num_ws, noise_mode=noise_mode)

        grid_videos, grid_images = generate_visualize_videos(lstm_ema, G.synthesis, **grid_args)
        grid_videos = ((grid_videos*0.5+0.5)*255).clamp(0,255).to(torch.uint8)
        grid_video_to_save = convert_batch_videos_to_grid(grid_videos, grid_size)

        save_image_grid(grid_images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=(grid_size[0], grid_size[1]))
        torchvision.io.write_video(os.path.join(run_dir, 'fake_init.mp4'), grid_video_to_save, fps=visualize_args.fps, video_codec='h264', options={'crf': '10'})

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            # real_image
            batch_real_imgs, batch_real_ts = next(training_set_iterator)
            phase_real_imgs = batch_real_imgs.to(torch.float32).to(device) / 127.5 - 1 # [b (t c) h w]
            phase_real_imgs = phase_real_imgs.split(batch_gpu)
            phase_real_ts = batch_real_ts.unsqueeze(-1).to(device).to(torch.float32)
            phase_real_ts = phase_real_ts.split(batch_gpu)

            #real_t = real_t_delta.view(*real_t_delta.shape, 1, 1).repeat(1, 1, *phase_real_imgs.shape[-2:]).to(device).to(torch.float32)
            #phase_real_imgs = torch.cat([phase_real_imgs, real_t], dim=1).split(batch_gpu)

            # gen_z
            batch_tmp = batch_size // num_gpus
            all_gen_zc = torch.randn([len(phases) * batch_tmp, content_dim], device=device)
            all_gen_zc = [phase_gen_zc.split(batch_gpu) for phase_gen_zc in all_gen_zc.split(batch_tmp)]

        # Execute training phases.
        for phase, phase_gen_zc in zip(phases, all_gen_zc):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_ts, gen_zc) in enumerate(zip(phase_real_imgs, phase_real_ts, phase_gen_zc)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_imgs=real_img, real_ts=real_ts, gen_zc=gen_zc, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
        
        # Update lstm_ema
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(lstm_ema.parameters(), lstm.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(lstm_ema.buffers(), lstm.buffers()):
            b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size * loss.n_sparse
        batch_idx += 1

        # Execute ADA heuristic
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image and video snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            grid_videos, grid_images = generate_visualize_videos(lstm_ema, G.synthesis, **grid_args)
            save_image_grid(grid_images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(grid_size[0], grid_size[1]))
            grid_videos = ((grid_videos*0.5+0.5)*255).clamp(0,255).to(torch.uint8)
            grid_video_to_save = convert_batch_videos_to_grid(grid_videos, grid_size)
            torchvision.io.write_video(os.path.join(run_dir, f'fake{cur_nimg//1000:06d}.mp4'), grid_video_to_save, fps=visualize_args.fps, video_codec='h264', options={'crf': '10'})

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        snapshot_modules = [
            ('G', G),
            ('D', D),
            ('lstm', lstm),
            ('lstm_ema', lstm_ema),
            ('augment_pipe', augment_pipe)
        ]
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            DDP_CONSISTENCY_IGNORE_REGEX = r'.*\.(w_avg|latent_avg|embeds.*\.weight|num_batches_tracked|running_mean|running_var)'
            for name, module in snapshot_modules:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=DDP_CONSISTENCY_IGNORE_REGEX)
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, lstm=snapshot_data['lstm_ema'], lstm_kwargs=lstm_kwargs, G=snapshot_data['G'], 
                    G_kwargs={'noise_mode': noise_mode}, dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------

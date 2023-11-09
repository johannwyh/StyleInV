# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
from urllib.parse import urlparse
import numpy as np
import torch
import dnnlib
from einops import rearrange
#----------------------------------------------------------------------------

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def trace_calculator(sigma, sigma_v):
    sigma_t = torch.from_numpy(sigma)
    sigma_v_t = torch.from_numpy(sigma_v)
    sqrt_trace = trace_sqrt_product(sigma_t, sigma_v_t)
    trace = torch.trace(sigma_t + sigma_v_t) - 2.0 * sqrt_trace
    return trace.item()

class MetricOptions:
    def __init__(self, SI=None, SI_kwargs={}, lstm=None, lstm_kwargs={}, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None,
                       progress=None, cache=True, gen_dataset_kwargs={}, generator_as_dataset=False):
        assert 0 <= rank < num_gpus
        self.SI                       = SI
        self.SI_kwargs                = dnnlib.EasyDict(SI_kwargs)
        self.lstm                     = lstm
        self.lstm_kwargs              = dnnlib.EasyDict(lstm_kwargs)
        self.G                        = G
        self.G_kwargs                 = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs           = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus                 = num_gpus
        self.rank                     = rank
        self.device                   = device if device is not None else torch.device('cuda', rank)
        self.progress                 = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache                    = cache
        self.gen_dataset_kwargs       = gen_dataset_kwargs
        self.generator_as_dataset     = generator_as_dataset

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            if urlparse(url).path.endswith('.pkl'):
                _feature_detector_cache[key] = pickle.load(f).eval().to(device)
            else:
                _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items: int):
        assert (self.num_items is None) or (cur_items <= self.num_items), f"Wrong `items` values: cur_items={cur_items}, self.num_items={self.num_items}"
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_dataset(
    opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64,
    data_loader_kwargs=None, max_items=None, temporal_detector=False, use_image_dataset=False,
    feature_stats_cls=FeatureStats, **stats_kwargs):

    dataset_kwargs = opts.dataset_kwargs
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs,
                    stats_kwargs=stats_kwargs, feature_stats_cls=feature_stats_cls.__name__)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'sinv-{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return feature_stats_cls.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)

    stats = feature_stats_cls(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)
    for images in dataloader:
        # images = batch # [b, c, t, h, w] or [b, c, h, w]

        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])
        features = detector(images.to(opts.device), **detector_kwargs)

        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic

    return stats

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_generator(
    opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size: int=16,
    batch_gen=None, jit=False, temporal_detector=False, num_video_frames: int=16,
    feature_stats_cls=FeatureStats, subsample_factor: int=1, **stats_kwargs):

    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    if opts.lstm is not None:
        lstm = copy.deepcopy(opts.lstm).eval().requires_grad_(False).to(opts.device)
        zc_dim = 512
    else:
        SI = copy.deepcopy(opts.SI).eval().requires_grad_(False).to(opts.device)
        zc_dim = SI.mapping_opts.content_dim
        if SI.mapping_opts.motion.type == 'motion_and_pe':
            zm_dim = SI.mapping_opts.motion.motion_dim
        elif SI.mapping_opts.motion.type == 'acyclic_pe':
            zm_dim = None
        clip = SI.clip_len
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Image generation func.
    def run_generator_lstm(zc, nframes):
        wc = G.mapping(zc, None)
        num_ws = wc.shape[1]
        wc = wc[:, 0, :]
        styles, _, _ = lstm(wc, nframes) # ((b t) c)
        styles_b = styles.unsqueeze(1).repeat([1, num_ws, 1]) # ((b t) n c)
        img = G.synthesis(styles_b, **opts.G_kwargs)

        if temporal_detector:
            img = rearrange(img, '(b t) c h w -> b c t h w', t=nframes)
        
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    def run_generator(zc, zm, ts):
        wc = G.mapping(zc, None)
        num_ws = wc.shape[1]
        x = G.synthesis(wc, noise_mode='const')
        wc = wc[:, 0, :]
        styles = SI(x, wc, zm, ts, run_parallel=True, return_temporal_style=False)
        styles_b = styles.unsqueeze(1).repeat([1, num_ws, 1])
        img = G.synthesis(styles_b, **opts.G_kwargs)

        if temporal_detector:
            img = rearrange(img, '(t b) c h w -> b c t h w', t=ts.shape[1])

        # img = torch.nn.functional.interpolate(img, size=(img.shape[2], 128, 128), mode='trilinear', align_corners=False) # downsample
        # img = torch.nn.functional.interpolate(img, size=(img.shape[2], 256, 256), mode='trilinear', align_corners=False) # upsample

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        t = torch.zeros([batch_gen, G.cfg.sampling.num_frames_per_video], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c, t], check_trace=False)

    # Initialize.
    stats = feature_stats_cls(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            if opts.lstm is not None:
                zc = torch.randn([batch_gen, zc_dim], device=opts.device)
                images.append(run_generator_lstm(zc, num_video_frames)) 
            else:   
                zc = torch.randn([batch_gen, zc_dim], device=opts.device)
                zm = None if zm_dim is None else torch.randn([batch_gen, zm_dim], device=opts.device) 
                max_frame_id = (num_video_frames - 1) * subsample_factor
                max_t_stamp = max_frame_id * (1.0 / (clip - 1))
                ts = torch.linspace(0, max_t_stamp, steps=num_video_frames).view(num_video_frames, 1).unsqueeze(0).to(opts.device)
                ts = ts.repeat([batch_gen, 1, 1])
                images.append(run_generator(zc, zm, ts))
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def rewrite_opts_for_gen_dataset(opts):
    """
    Updates dataset arguments in the opts to enable the second dataset stats computation
    """
    new_opts = copy.deepcopy(opts)
    new_opts.dataset_kwargs = new_opts.gen_dataset_kwargs
    new_opts.cache = False

    return new_opts

#----------------------------------------------------------------------------
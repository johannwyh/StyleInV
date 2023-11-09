import os
import os.path as osp
import numpy as np
import zipfile
import PIL.Image
from PIL import Image
from PIL import ImageFile
import json
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
import random
import pickle
import dnnlib
from dnnlib.util import EasyDict
from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from sampling import BetaSampling, RandomSampling
from copy import deepcopy
import pyspng
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def resize_crop(video, resolution):
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop
    Args
        video: a tensor of shape [t, c, h, w] in {0, ..., 255}
        resolution: an int
        crop_mode: 'center', 'random'
    Returns
        a processed video of shape [c, t, h, w]
    """
    _, _, h, w = video.shape

    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w >= h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    video = video[:, :, cropsize[1]:cropsize[3],  cropsize[0]:cropsize[2]]
    video = F.interpolate(video, size=resolution, mode='bilinear', align_corners=False)

    video = video.permute(1, 0, 2, 3).contiguous()  # [c, t, h, w]
    return video


def resize_crop_img(image, resolution):
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop
    Args
        image: a tensor of shape [c h w] in {0, ..., 255}
        resolution: an int
    Returns
        a processed img of shape [c, h, w]
    """
    # [c h w]
    _, h, w = image.shape
    image = torch.from_numpy(image).unsqueeze(dim=0)  # 1, c, h, w
    
    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w >= h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    image = image[:, :, cropsize[1]:cropsize[3], cropsize[0]:cropsize[2]]
    #print("Before Interpolate, ", type(image), image.shape)

    image = F.interpolate(image, size=resolution, mode='bilinear', align_corners=False)
    #print("After Interpolate, ", type(image), image.shape)

    return image.squeeze(dim=0).numpy()  # c, h, w

class StyleInVDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,
                 nframes=16,  # number of frames for each video.
                 stride=1, # temporal stride among frames
                 flex_sampling='none', # Flex sampling mode, in ['none', 'allow_short', 'full']
                 random_offset=True, # When True, the start frame can by any within a clip,
                 return_one=False, # True for evaluating FID
                 random_select=True, # Whether to randomly sample a video then clip for return_one mode
                 only_first=-1, # The number of first frames that is accounted in the dataset for each video sample
                 return_vid=False,  # True for evaluating FVD
                 sampling_cfg={},
                 timestamp_type='normalized',
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):

        valid_timestamp_types = ['normalized', 'index']
        valid_flex_sampling_type = ['none', 'allow_short', 'full']
        assert timestamp_type in valid_timestamp_types, f'{timestamp_type} not in {valid_timestamp_types}'
        if flex_sampling == False:
            flex_sampling = 'none'
        assert flex_sampling in valid_flex_sampling_type, f'{flex_sampling} not in {valid_flex_sampling_type}'

        self.timestamp_type = timestamp_type

        # return_f0
        self.sampling_cfg = deepcopy(sampling_cfg)
        self.sampling_cfg.use_fractional_t = False
        if self.sampling_cfg.type == 'random':
            self.sampler = RandomSampling(self.sampling_cfg)
        elif self.sampling_cfg.type == 'beta':
            self.sampler = BetaSampling(self.sampling_cfg)

        # adopted from DIGAN
        self._path = path
        self._zipfile = None
        self.flex_sampling = flex_sampling
        self.only_first = only_first
        if self.flex_sampling in ['allow_short', 'full']:
            self.min_frames_train = (self.sampling_cfg.num_frames_per_video - 1) * stride + 1
        else:
            self.min_frames_train = (nframes - 1) * stride + 1
        vids, imgs = [], []
        
        all_entries = list(os.walk(self._path, followlinks=True))

        for dir, _, files in all_entries:
            this_vid = []
            for frame in files:
                if is_image_file(frame):
                    this_vid.append(os.path.join(dir, frame))
            this_vid = sorted(this_vid)
            
            # specify only_first=16, if follows StyleGAN-V way to calculate FID
            if self.only_first > 0:
                this_vid = this_vid[:self.only_first]
            
            for _ in this_vid:
                imgs.append(_)
            #imgs = imgs + this_vid

            if return_vid: # fvd dataset
                if len(this_vid) >= (nframes-1) * stride + 1:
                    vids.append(this_vid)
            elif return_one: # fid dataset
                if len(this_vid) > 0:
                    vids.append(this_vid)
            else: # training dataset
                if len(this_vid) >= self.min_frames_train:
                    vids.append(this_vid)
                
        if len(vids) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + path + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        imgs = sorted(imgs)
        vids = sorted(vids, key=lambda x: x[0])

        self.imgs = imgs
        self.vids = vids
        self.nframes = nframes

        self.img_resolution = resolution
        dataset_resolution = PIL.Image.open(self.vids[0][0]).size[0]
        if self.img_resolution is None:
            self.img_resolution, self.apply_resize = dataset_resolution, False
        else:
            self.apply_resize = (dataset_resolution != self.img_resolution)
        
        self.stride = stride
        self.xflip = super_kwargs["xflip"]
        self._total_size = len(self.vids) * 2 if self.xflip else len(self.vids)
        self._total_imgs = len(self.imgs)
        self.n_videos = len(self.vids)
        self.total_frames = self._total_imgs

        # special args for metric evaluation
        self.return_vid = return_vid
        self.return_one = return_one
        self.random_offset = random_offset
        self.random_select = random_select

        if (not self.return_vid) and (not self.return_one):
            assert self.sampling_cfg.max_num_frames == self.nframes, f'Sampling maximum frames {self.sampling_cfg.max_num_frames} should equal to dataset clip length {self.nframes}'
        
        self.shuffle_indices = list(range(self._total_size))
        self.shuffle_indices_img = list(range(self._total_imgs))

        self.to_tensor = transforms.ToTensor()
        
        random.shuffle(self.shuffle_indices)
        random.shuffle(self.shuffle_indices_img)

        self.name = os.path.splitext(os.path.basename(self._path))[0]
        
        super().__init__()
        
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()


    def _open_file(self, fname):
        return open(fname, 'rb')

    def _load_img_from_path(self, path):
        if os.path.splitext(path)[1].lower() == '.png':
            with self._open_file(path) as f:
                try:
                    img = pyspng.load(f.read())
                except:
                    img = np.array(PIL.Image.open(f))
        else:
            with self._open_file(path) as f:
                img = np.array(PIL.Image.open(f))
        if img.ndim == 2:
            img = img[:, :, np.newaxis] # HW => HWC
        img = img.transpose(2, 0, 1) # HWC => CHW
        assert img.dtype == np.uint8
        return img

    def load_image(self, index):
        if self.random_select:
            index = self.shuffle_indices[index % self._total_size]
            clip = self.vids[index - self._total_size // 2] if self.xflip else self.vids[index]
            len_vid = len(clip)
            idx = np.random.randint(0, len_vid)
            img_path = clip[idx]
        else:
            index = self.shuffle_indices_img[index % self._total_imgs]
            img_path = self.imgs[index]
        img = self._load_img_from_path(img_path)
        if img.shape[1] != self.img_resolution:
            img = resize_crop_img(img, self.img_resolution)
        if self.xflip and index >= self._total_size // 2:
            img = img[:, :, ::-1]
        return img.copy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.return_one:
            return self.load_image(index)

        index = self.shuffle_indices[index % self._total_size]
        clip = self.vids[index - self._total_size // 2] if self.xflip else self.vids[index]
        len_vid = len(clip)
        
        if self.random_offset:
            if self.return_vid: # fvd dataset
                max_idx = len_vid - (self.nframes - 1) * self.stride
            else: # training dataset
                if self.flex_sampling == 'none':
                    max_idx = len_vid - self.min_frames_train + 1
                elif self.flex_sampling == 'allow_short':
                    max_idx = max(0, len_vid - self.nframes) + 1
                elif self.flex_samplint == 'full':
                    max_idx = len_vid - self.min_frames_train + 1
            start = np.random.randint(0, max_idx)
        else:
            start = 0
            
        end = start + (self.nframes - 1) * self.stride + 1
        clip = clip[start:end:self.stride]

        if self.return_vid:
            # ignore xflip for return_vid
            vid = np.stack([self._load_img_from_path(clip[i]).copy() for i in range(self.nframes)], axis=0)
            if self.xflip and index >= self._total_size // 2:
                vid = vid[:, :, :, ::-1]
            if self.apply_resize:
                return resize_crop(torch.from_numpy(vid), resolution=self.img_resolution).numpy()
            return rearrange(vid, 't c h w -> c t h w').copy()

        frames = self.sampler.sample_one(max_limit=len(clip)) # [t], [0, 1]

        imgs = []
        for frame_id in frames:
            frame_idx = round(frame_id * (self.nframes - 1))
            path = clip[frame_idx]
            img = self._load_img_from_path(path).astype(np.float32)
            if self.apply_resize:
                img = resize_crop_img(img, self.img_resolution)
            if self.xflip and index >= self._total_size // 2:
                img = img[:, :, ::-1]
            imgs.append(img)
        
        Ts = []
        for i_frame in range(0, len(frames)):
            t_index = frames[i_frame]
            Ts.append(t_index)
        
        return np.concatenate(imgs, axis=0).copy(), np.array(Ts)

    def __len__(self):
        if self.return_one:
            return self._total_imgs
        else:
            return self._total_size

# def video_to_image_dataset_kwargs(video_dataset_kwargs: dnnlib.EasyDict) -> dnnlib.EasyDict:
#     """Converts video dataset kwargs to image dataset kwargs"""
#     return dnnlib.EasyDict(
#         class_name='training.dataset_styleinv.StyleInVDataset',
#         path=video_dataset_kwargs.path,
#         xflip=False,
#         resolution=video_dataset_kwargs.resolution,
#         return_one=True,
#         sampling_cfg=video_dataset_kwargs.sampling_cfg, # no use, just to make the code run
#         # Explicitly ignoring the max size, since we are now interested
#         # in the number of images instead of the number of videos
#         # max_size=video_dataset_kwargs.max_size,
#     )
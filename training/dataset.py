# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        video_balance   = False, # Whether to balance the entries of each video clip
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.video_balance = video_balance

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path, followlinks=True) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')
                       
        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        if self.video_balance:
            hierarchy_fnames = []
            current_vid, vid_name = [self._image_fnames[0]], os.path.split(self._image_fnames[0])[0]
            for i_img in range(1, len(self._image_fnames)):
                vid_name_i = os.path.split(self._image_fnames[i_img])[0]
                if vid_name_i == vid_name:
                    current_vid.append(self._image_fnames[i_img])
                else:
                    hierarchy_fnames.append(current_vid)
                    current_vid = [self._image_fnames[i_img]]
                    vid_name = vid_name_i
            hierarchy_fnames.append(current_vid)
            self._hierarchy_fnames = hierarchy_fnames
            self.n_videos = len(self._hierarchy_fnames)
            self.shuffle_indices_vid = list(range(self.n_videos))
            random.shuffle(self.shuffle_indices_vid)

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        if not self.video_balance:
            fname = self._image_fnames[raw_idx]
        else:
            vid_idx = self.shuffle_indices_vid[raw_idx % self.n_videos]
            video = self._hierarchy_fnames[vid_idx]
            frame_idx = np.random.randint(0, len(video))
            fname = video[frame_idx]
        
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

import torchvision.transforms as transforms
PSP_Transform_Dict = {
    'transform_gt_train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    'transform_source': None,
    'transform_test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    'transform_inference': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

def convert_image_list_to_video(image_fnames):
    hierarchy = []
    current_vid, vid_name = [image_fnames[0]], os.path.split(image_fnames[0])[0]
    for i_img in range(1, len(image_fnames)):
        vid_name_i = os.path.split(image_fnames[i_img])[0]
        if vid_name_i == vid_name:
            current_vid.append(image_fnames[i_img])
        else:
            hierarchy.append(current_vid)
            current_vid = [image_fnames[i_img]]
            vid_name = vid_name_i
    hierarchy.append(current_vid)
    n_videos = len(hierarchy)
    shuffle_indices_vid = list(range(n_videos))
    random.shuffle(shuffle_indices_vid)
    return hierarchy, shuffle_indices_vid

class PSPDataset(torch.utils.data.Dataset):

    def __init__(self, source_root, target_root, transform_type, label_nc=0, video_balance=False, video_unit=False, video_frame_max=128):
        self.name = "PSP_Inv"
        self.source_paths = sorted(make_dataset(source_root))
        self.target_paths = sorted(make_dataset(target_root))
        self.video_balance = video_balance
        self.video_unit = video_unit
        if video_unit:
            self.video_frame_max = video_frame_max

        if self.video_balance or self.video_unit:
            self.source_videos, self.source_vid_idx = convert_image_list_to_video(self.source_paths)
            self.target_videos, self.target_vid_idx = convert_image_list_to_video(self.target_paths)

        assert transform_type in ['train', 'test']
        if transform_type == 'train':
            self.target_transform = PSP_Transform_Dict['transform_gt_train']
        else:
            self.target_transform = PSP_Transform_Dict['transform_test']
        self.source_transform = PSP_Transform_Dict['transform_source']
        self.label_nc = label_nc

    def __len__(self):
        if self.video_balance or self.video_unit:
            return len(self.source_vid_idx)
        else:
            return len(self.source_paths)

    def return_video(self, index):
        frames = sorted(self.target_videos[index][:self.video_frame_max])
        video = [self.target_transform(PIL.Image.open(path).convert('RGB').copy()) for path in frames]
        return torch.stack(video)

    def __getitem__(self, index):
        if self.video_unit:
            return self.return_video(index)

        if not self.video_balance:
            from_path = self.source_paths[index]
            to_path = self.target_paths[index]
        else:
            # from
            from_vid = self.source_vid_idx[index]
            from_frame_index = np.random.randint(0, len(self.source_videos[from_vid]))
            from_path = self.source_videos[from_vid][from_frame_index]
            # to
            to_vid = self.target_vid_idx[index]
            to_frame_index = np.random.randint(0, len(self.target_videos[to_vid]))
            to_path = self.target_videos[to_vid][to_frame_index]

        from_im = PIL.Image.open(from_path)
        from_im = from_im.convert('RGB') if self.label_nc == 0 else from_im.convert('L')

        to_im = PIL.Image.open(to_path).convert('RGB')
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im

#----------------------------------------------------------------------------
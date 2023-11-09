import numpy as np
import torch

class RandomSampling:
    def __init__(self, cfg):
        self.type = 'random'
        self.cfg = cfg
        self.add_0 = cfg.add_0
        self.num_sample = cfg.num_sample
        
        assert cfg.num_frames_per_video == cfg.num_sample + (cfg.add_0 == True)

        self.max_num_frames = cfg.max_num_frames
        self.use_fractional_t = cfg.use_fractional_t

    def sample_one(self, max_limit=-1):
        # self.max_num_frames: corresponding to clip length
        # max_num_frames: can be current video length, to make use of short clips

        min_time_diff = self.num_sample - 1
        if max_limit > 0 and max_limit < self.max_num_frames:
            max_num_frames = max_limit
        else:
            max_num_frames = self.max_num_frames

        max_time_diff = min(max_num_frames - 1, self.cfg.get('max_dist', float('inf')))

        if type(self.cfg.get('total_dists')) in (list, tuple):
            time_diff_range = [d for d in self.cfg['total_dists'] if min_time_diff <= d <= max_time_diff]
        else:
            time_diff_range = range(min_time_diff, max_time_diff)

        time_diff: int = random.choice(time_diff_range)
        if self.use_fractional_t:
            offset = random.random() * (max_num_frames - time_diff - 1)
        else:
            offset = random.randint(0, max_num_frames - time_diff - 1)
        frames_idx = [offset]
        
        if self.num_sample > 1:
            frames_idx.append(offset + time_diff)

        if self.num_sample > 2:
            frames_idx.extend([(offset + t) for t in random.sample(range(1, time_diff), k=self.num_sample - 2)])

        frames_idx = sorted(frames_idx)

        if self.add_0:
            frames_idx = [0] + frames_idx

        return np.array(frames_idx).astype(np.float32) / (self.max_num_frames - 1)
    
    def sample(self, batch_size, device):
        # output: [b, t, 1]
        Ts = [torch.from_numpy(self.sample_one()).unsqueeze(0) for i in range(batch_size)]
        Ts = torch.cat(Ts, dim=0).unsqueeze(-1).to(device) # (b, t, 1)
        return Ts

class BetaSampling:
    def __init__(self, cfg):
        self.type = 'beta'
        self.cfg = cfg
        self.add_0 = cfg.add_0
        self.num_sample = 2
        self.dist1 = torch.distributions.beta.Beta(2., 1., validate_args=None)
        self.dist2 = torch.distributions.beta.Beta(1., 2., validate_args=None)
        self.max_num_frames = cfg.max_num_frames
        self.use_fractional_t = cfg.use_fractional_t

    def sample_one(self, max_limit=-1):
        if max_limit > 0 and max_limit < self.max_num_frames:
            max_num_frames = max_limit
        else:
            max_num_frames = self.max_num_frames

        a = self.dist1.sample().item()
        b = self.dist2.sample().item()
        if a > b:
            a, b = b, a

        if not self.use_fractional_t:
            a = (a * (max_num_frames - 1)).floor() / (self.max_num_frames - 1)
            b = (b * (max_num_frames - 1)).floor() / (self.max_num_frames - 1)
        else:
            a = (a * (max_num_frames - 1)) / (self.max_num_frames - 1)
            b = (b * (max_num_frames - 1)) / (self.max_num_frames - 1)
            
        if self.add_0:
            return 0, a, b
        else:
            return a, b
            
    def sample(self, batch_size, device):
        # output: [b, t, 1]
        Ts_12 = torch.cat([self.dist1.sample((batch_size, 1, 1)),
                        self.dist2.sample((batch_size, 1, 1))], dim=1).to(self.device)
        Ts_12 = torch.cat([Ts_12.min(dim=1, keepdim=True)[0], Ts_12.max(dim=1, keepdim=True)[0]], dim=1)
        if self.add_0:
            Ts = torch.cat([torch.zeros(batch_size, 1, 1).to(device), Ts_12], dim=1)
        else:
            Ts = Ts_12
        
        if not self.use_fractional_t:
            Ts = (Ts * self.max_num_frames).floor() / (self.max_num_frames - 1)

        return Ts
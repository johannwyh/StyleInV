import torch
from torch import nn
from auxiliary.config import psp_model_paths as model_paths
from auxiliary.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        #print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        facenet_ckpt = torch.load(model_paths['ir_se50'], map_location=torch.device('cpu'))
        self.facenet.load_state_dict(facenet_ckpt)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            #diff_input = y_hat_feats[i].dot(x_feats[i])
            #diff_views = y_feats[i].dot(x_feats[i])
            #id_logs.append({'diff_target': float(diff_target),
            #                'diff_input': float(diff_input),
            #                'diff_views': float(diff_views)})
            loss += 1 - diff_target
            #id_diff = float(diff_target) - float(diff_views)
            #sim_improvement += id_diff
            count += 1

        return loss / count

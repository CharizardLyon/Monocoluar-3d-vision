import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_higher_hrnet import get_pose_net

class HRNet3DPose(nn.module):
    def __init__(self, cfg, output_joints=21, feature_dim=2048):
        super(HRNet3DPose, self).__init__()

        self.backbone = get_pose_net(cfg, is_train=True)

        self.pool = nn.AdaptativeAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_joints * 3)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.mlp(x)
        return x.view(x.size(0), -1, 3)
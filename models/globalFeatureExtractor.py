import torch as th
import torch.nn as nn


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(GlobalFeatureExtractor, self).__init__()

        self.input_transform = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3 * 3, 1)
        )

        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.feature_transform_conv = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.feature_transform_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64)
        )

        self.feature_transform_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.global_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        x = x.permute(0, 2, 1)

        transform_3x3 = self.input_transform(x)
        transform_3x3 = transform_3x3.max(dim=-1)[0].view(batch_size, 3, 3)
        x = th.bmm(transform_3x3, x)

        local_feat_64 = self.mlp1(x)

        transform_64 = self.feature_transform_conv(local_feat_64)
        transform_64 = self.feature_transform_pool(transform_64).squeeze(-1)
        transform_64 = self.feature_transform_fc(transform_64).view(batch_size, 64, 64)

        identity = th.eye(64, device=x.device).view(1, 64, 64)
        transform_64 = transform_64 + identity

        local_feat_64 = th.bmm(transform_64, local_feat_64)

        local_feat = self.mlp2(local_feat_64)
        global_feat = self.global_pooling(local_feat).view(batch_size, 1024)

        return global_feat
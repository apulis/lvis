import torch
import torch.nn as nn

from ..utils import kaiming_init


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.sigmoid = nn.Sigmoid()
        self.flatten = Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        assert 'avg' in pool_types
        self.pool_types = pool_types
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.init_weights()

    def forward(self, x):
        # x: RPN features corresponding total pyramid lavels.
        # x: ([B, C, H, W], )  (tuple)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = torch.sum(
                    torch.stack([self.avg_pool(feat) for feat in x], dim=0), 0)
                avg_pool = self.flatten(avg_pool)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = torch.sum(
                    torch.stack([self.max_pool(feat) for feat in x], dim=0), 0)
                max_pool = self.flatten(max_pool)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # (B, C) -> (B, C, 1, 1)
        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return avg_pool, scale

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_init(m)

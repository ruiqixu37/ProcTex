import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].

    Code taken from https://github.com/threedle/text2mesh
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self.register_buffer('_B', torch.stack(B_sort))

    def forward(self, x):
        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        res = x @ self._B.to(x.device)
        res *= 2 * np.pi
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)
    
def positional_encoding(x, degree=1):
    """
    Apply positional encoding to the input tensor x.

    :param x: Input tensor of shape (batch_size, 2 | 3).
    :param degree: Degree of positional encoding.
    :return: Positional encoded tensor.
    """
    if degree < 1:
        return x

    pe = [x]
    for d in range(1, degree + 1):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2.0**d * math.pi * x))
    return torch.cat(pe, dim=-1)

class UVDisplacementNetwork(nn.Module):
    def __init__(
        self, input_dim=2, output_dim=2, hidden_dim=256, num_layers=8, degree=1, procedural_param_count=0
    ):
        super(UVDisplacementNetwork, self).__init__()

        self.procedural_parameters = None
        self.degree = degree
        self.input_dim = input_dim + procedural_param_count
        # * (
        #     2 * degree + 1
        # ) + procedural_param_count

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        # layers.append(nn.Tanh())
        # layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def update_procedural_parameters(self, parameters):
        self.procedural_parameters = parameters

    def forward(self, x):
        # x = positional_encoding(x, self.degree)
        # concatenate procedural parameters with x
        if self.procedural_parameters is not None:
            repeated_procedural_parameters = self.procedural_parameters.repeat(x.shape[0], 1)
        else:
            repeated_procedural_parameters = torch.zeros(x.shape[0], 0, device=x.device)
        x = torch.cat([x, repeated_procedural_parameters], dim=-1)
        out = self.model(x)
        return out
    

class PointNet(nn.Module):
    def __init__(self, feature_dim=128, num_mlp1_layers=1, num_mlp2_layers=1):
        super(PointNet, self).__init__()

        self.feature_dim = feature_dim

        mlp1_layers = [
            nn.Linear(3, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.ReLU()
        ]
        for _ in range(num_mlp1_layers):
            mlp1_layers.append(nn.Linear(feature_dim, feature_dim))
            mlp1_layers.append(nn.ReLU())

        self.mlp1 = nn.Sequential(*mlp1_layers)

        mlp2_layers = [
            nn.Linear(feature_dim * 2, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU()
        ]
        for _ in range(num_mlp2_layers):
            mlp2_layers.append(nn.Linear(feature_dim, feature_dim))
            mlp2_layers.append(nn.ReLU())

        mlp2_layers.append(nn.Linear(feature_dim, 3))
        self.mlp2 = nn.Sequential(*mlp2_layers)

    def forward(self, x):
        feat_local = self.mlp1(x)  # (N, F)
        global_feat = feat_local.max(dim=0, keepdim=True)[0].repeat(x.size(0), 1)  # (N, F)
        feat_cat = torch.cat([feat_local, global_feat], dim=1)  # (N, 2F)
        out = self.mlp2(feat_cat)  # (N, 3)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

def gs_activate_head(inputs, residual):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    fmap = inputs #B,N,C    
    means =  fmap[ :, :, :3]
    color = fmap[:, :, 3:6]
    opacity = fmap[:, :, 6:7]
    scale = fmap[:,  :, 7:10]
    rotation = fmap[:, :, 10:14]

    fmap = residual #B,N,C    
    means = means + fmap[ :, :, :3] * 0.1
    color = color + fmap[:, :, 3:6] * 0.1
    opacity = opacity + fmap[:, :, 6:7] * 0.1
    scale = scale + fmap[:,  :, 7:10] * 0.01
    rotation = F.normalize(rotation + fmap[:, :, 10:14] * 0.01)
    
    #color = torch.sigmoid(color)
    #opacity = torch.sigmoid(opacity) 
    #scale =  F.softplus(scale)
    #rotation = F.normalize(rotation, dim=-1)

    pts3d = torch.concat([means,color,opacity,scale,rotation],dim=-1)


    return pts3d


class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        layers = []
        last_ch = in_channels
        for out_ch in mlp_channels:
            layers.append(nn.Conv1d(last_ch, out_ch, 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, N]
        return self.mlp(x)

class PointNetGSFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 14  # 3+3+1+3+4
        self.abstraction1 = PointNetSetAbstraction(self.input_dim, [64, 128])
        self.abstraction2 = PointNetSetAbstraction(128, [128, 256])
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 14, 1)  # Predict refined Gaussian parameters
        )

    def forward(self, inputs):
        # x: [B, N, 14]
        x = inputs.permute(0, 2, 1)  # [B, 14, N]
        x = self.abstraction1(x)  # [B, 128, N]
        x = self.abstraction2(x)  # [B, 256, N]
        x = self.fusion_mlp(x)    # [B, 14, N]
        x = x.permute(0, 2, 1)    # [B, N, 14]
        x = gs_activate_head(inputs, x)
        embed()

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs[0,:,:3].detach().cpu().numpy())
        # Save to .ply
        o3d.io.write_point_cloud('input.ply', pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x[0,:,:3].detach().cpu().numpy())
        # Save to .ply
        o3d.io.write_point_cloud('x.ply', pcd)
        return x

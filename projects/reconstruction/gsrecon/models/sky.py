import numpy as np
import math
import open3d as o3d
from gsrecon.models.projector import Projector
from gsplat.rendering import rasterization
import torch
import torch.nn as nn
from IPython import embed
from typing import Optional, Tuple, Set


def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = (i / float(samples - 1)) -1   # y goes from 0 to -1
        radius = math.sqrt(1 - y ** 2)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))

    return points

def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def k_nearest_sklearn(x: torch.Tensor, k: int):
    """
    Find k-nearest neighbors using sklearn's NearestNeighbors.
    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()

    # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def scale_intrinsics(
    Ks: torch.Tensor,  # [N, 4, 4]
    downsample_rate: float
) -> torch.Tensor:
    Ks_scaled = Ks.clone()
    Ks_scaled[:, 0, 0] /= downsample_rate  # fx
    Ks_scaled[:, 1, 1] /= downsample_rate  # fy
    Ks_scaled[:, 0, 2] /= downsample_rate  # cx
    Ks_scaled[:, 1, 2] /= downsample_rate  # cy
    return Ks_scaled


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_freqs).float()  # [1, 2, 4, ..., 2^(num_freqs-1)]

    def forward(self, x):
        """
        x: [..., D], typically D=3 for xyz
        returns: [..., D * (include_input + 2*num_freqs)]
        """
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)
    

class NeRF(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256, pe_freqs=10, n_layers=6):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=pe_freqs)
        pe_dim = 3 * (1 + 2 * pe_freqs)
        
        self.input_layer = nn.Linear(feature_dim + pe_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)])
        self.output_layer = nn.Linear(hidden_dim, 3)
        
    def forward(self, image_feature, point_xyz):
        pe_xyz = self.pe(point_xyz)
        x = torch.cat([image_feature, pe_xyz], dim=-1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        color = torch.sigmoid(self.output_layer(x))
        return color




class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert in_dim > 0, "Input dimension must be positive"

        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections or ()
        self._skip_connections: Set[int] = set(self.skip_connections)
        self.activation = activation
        self.out_activation = out_activation

        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Builds the torch MLP with optional skip connections."""
        layers = []

        for i in range(self.num_layers):
            if i == 0:
                in_features = self.in_dim
            elif i in self._skip_connections:
                in_features = self.layer_width + self.in_dim
            else:
                in_features = self.layer_width

            out_features = self.out_dim if i == self.num_layers - 1 else self.layer_width
            layers.append(nn.Linear(in_features, out_features))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies MLP to the input."""
        input_copy = x
        for i, layer in enumerate(self.layers):
            if i in self._skip_connections:
                x = torch.cat([input_copy, x], dim=-1)
            x = layer(x)
            if i < len(self.layers) - 1 and self.activation is not None:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


class SkyGaussian(nn.Module):
    def __init__(self,resolution = 300 ,radius = 50,center=np.array([0,0,0])):
        super().__init__()
        self.resolution = resolution
        self.radius = radius
        self.center = center
        self.projector = Projector()

        self.bg_field = MLP(
            in_dim= 3,#9,
            num_layers=2,
            layer_width=64,
            out_dim=6,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            )


        # self.bg_field = NeRF()

        num_background_points = self.resolution ** 2
        xyz = fibonacci_sphere(num_background_points)
        xyz = np.array(xyz) * self.radius
        sky_pnt = xyz.astype(np.float32)
        sky_pnt += self.center
        bg_distances, _ = k_nearest_sklearn(torch.from_numpy(sky_pnt), 3)
        bg_distances = torch.from_numpy(bg_distances)
        avg_dist = bg_distances.mean(dim=-1, keepdim=True)

        bg_scales = torch.log(avg_dist.repeat(1, 3)) #torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self.register_buffer("bg_scales",bg_scales)
        self.register_buffer("bg_pcd", torch.tensor(sky_pnt))
        num_bg_points = sky_pnt.shape[0]
        self.register_buffer("bg_opacity", torch.ones(num_bg_points, 1))
        quat = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float32).repeat(num_bg_points, 1)
        self.register_buffer("bg_quat", quat)


    def _get_background_color(self, source_images, source_extrinsics, intrinsics, downsample=1):
        ## add projection mask to decrease the computational complexity
        intrinsics = scale_intrinsics(intrinsics, downsample)
        sampled_feat,proj_mask,normalized_pixel_locations = self.projector.compute(xyz = self.bg_pcd.reshape(-1,3), 
                                        train_imgs = source_images.squeeze(0),        #N,C,H,W        
                                        train_cameras = source_extrinsics,     #N,4,4
                                        train_intrinsics= intrinsics,       #N,4,4
                                        )
        
        sampled_bg_pcd = self.bg_pcd[proj_mask]
        

        #sampled_feat = sampled_feat.mean(dim=1)

        # background_rgb = self.bg_field(sampled_feat, sampled_bg_pcd).float()
        background_feat = self.bg_field(sampled_feat.view(-1,3)).float()
        background_rgb, background_scale_res = background_feat.split([3,3],dim=-1)
        background_scale_res = torch.zeros_like(background_rgb)
        return background_rgb, proj_mask, background_scale_res
        # return sampled_feat.view(-1,3), proj_mask, background_scale_res
    

    def forward_with_new_pose(self, images, extrinsics, intrinsics, extrinsics_, intrinsics_, downsample=1):
        S = extrinsics.shape[0]
        intrinsics_4x4 = torch.eye(4).unsqueeze(0).repeat(S, 1, 1).to(device=intrinsics.device)
        intrinsics_4x4[:,:3, :3] = intrinsics
        
        background_feat,proj_mask, background_scale_res = self._get_background_color(
                                                                                    source_images=images,
                                                                                    source_extrinsics= extrinsics,
                                                                                    intrinsics= intrinsics_4x4,
                                                                                    downsample = downsample
                                                                                    )


        H, W = images.shape[-2:]

        chunk_size = 4 
        chunked_renders = []
        S_ = extrinsics_.shape[0]
        for start in range(0, S_, chunk_size):
            end = min(start + chunk_size, S_)
            bg_render, _, _  = rasterization(
                        means=self.bg_pcd[proj_mask],       
                        quats=self.bg_quat[proj_mask],           
                        scales=torch.exp(self.bg_scales)[proj_mask] + background_scale_res,           
                        opacities=self.bg_opacity.squeeze(-1)[proj_mask],        
                        colors=background_feat,              
                        viewmats=extrinsics_[start:end,...],     # (chunk, 4, 4)
                        Ks=intrinsics_[start:end,...],        # (chunk, 3, 3)
                        width=W,
                        height=H
                    )
            # #
            points = self.bg_pcd[proj_mask]


            color = background_feat
            # import open3d as o3d
            # points_np = points.cpu().numpy()
            # colors_np = color.cpu().numpy()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_np)
            # pcd.colors = o3d.utility.Vector3dVector(colors_np)  # Should be in [0, 1] range
            # o3d.io.write_point_cloud("output.ply", pcd)
            #T.ToPILImage()(bg_render[0].permute(2,0,1).detach().cpu().clamp(0, 1)).save(f"test.png")


            chunked_renders.append(bg_render)
        bg_render = torch.cat(chunked_renders, dim=0)  # (B, S, 3, H, W)
        return bg_render


    def forward(self, images, extrinsics, intrinsics,downsample=1):
        S = extrinsics.shape[0]
        intrinsics_4x4 = torch.eye(4).unsqueeze(0).repeat(S, 1, 1).to(device=intrinsics.device)
        intrinsics_4x4[:,:3, :3] = intrinsics

        background_feat,proj_mask, background_scale_res = self._get_background_color(
                                                                                    source_images=images,
                                                                                    source_extrinsics= extrinsics,
                                                                                    intrinsics= intrinsics_4x4,
                                                                                    downsample = downsample
                                                                                    )

        H, W = images.shape[-2:]

        chunk_size = 4 
        chunked_renders = []
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            bg_render, _, _  = rasterization(
                        means=self.bg_pcd[proj_mask],       
                        quats=self.bg_quat[proj_mask],           
                        scales=torch.exp(self.bg_scales)[proj_mask] + background_scale_res,           
                        opacities=self.bg_opacity.squeeze(-1)[proj_mask],        
                        colors=background_feat,              
                        viewmats=extrinsics[start:end,...],     # (chunk, 4, 4)
                        Ks=intrinsics[start:end,...],        # (chunk, 3, 3)
                        width=W,
                        height=H
                    )
            # #
            points = self.bg_pcd[proj_mask]
            # color = background_feat
            # import open3d as o3d
            # points_np = points.cpu().numpy()
            # colors_np = color.cpu().numpy()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_np)
            # pcd.colors = o3d.utility.Vector3dVector(colors_np)  # Should be in [0, 1] range
            # o3d.io.write_point_cloud("output.ply", pcd)
            #T.ToPILImage()(bg_render[0].permute(2,0,1).detach().cpu().clamp(0, 1)).save(f"test.png")
            chunked_renders.append(bg_render)
        bg_render = torch.cat(chunked_renders, dim=0)  # (B, S, 3, H, W)
        return bg_render
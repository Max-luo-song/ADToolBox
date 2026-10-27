import torch
import torch.nn as nn
from einops import rearrange

class ConvUpdater(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        base_filters: int, 
        output_dim: int, 
        n_blocks: int, 
        kernel_size: int, 
        increasefilter_gap: int, 
        groups: int, 
        use_norm: bool, 
        use_dropout: bool
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_filters = base_filters
        self.output_dim = output_dim

        self.n_blocks = n_blocks
        
        self.first_block_conv = nn.Conv1d(in_channels=input_dim, out_channels=base_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.first_block_norm = nn.GroupNorm(1, base_filters)
        self.first_block_relu = nn.ReLU(inplace=False)
        out_channels = base_filters
                
        self.basicblock_list = nn.ModuleList()
        for i_block in range(n_blocks):
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False

            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters*2**((i_block-1)//increasefilter_gap))
                if (i_block % increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups),
                nn.GroupNorm(1, out_channels) if use_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(0.1) if use_dropout else nn.Identity()
            )
            self.basicblock_list.append(tmp_block)

        self.final_norm = nn.GroupNorm(1, out_channels)
        self.final_relu = nn.ReLU(inplace=False)
        self.dense = nn.Linear(out_channels, output_dim)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x = rearrange(x, "b t n c -> (b n) c t")
        out = self.first_block_conv(x)
        out = self.first_block_relu(out)

        for i_block in range(self.n_blocks):
            net = self.basicblock_list[i_block]
            out = net(out)
        
        out = self.final_relu(out)
        out = rearrange(out, "(b n) c t -> b t n c", b=B)
        
        delta = self.dense(out)
        return delta


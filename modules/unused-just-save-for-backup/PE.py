class PatchEmbed(torch.nn.Module):

    def __init__(self, in_dim: int = 4, dim: int = 96, num_stages: int = 2, nempty: bool = True, **kwargs):
        super().__init__()
        self.num_stages = num_stages
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.mlps = torch.nn.ModuleList(
            [MLP(in_dim if i == 0 else channels[i-1], channels[i], channels[i]) for i in range(self.num_stages)])
        self.norm = torch.nn.LayerNorm(channels[-2])
        self.downsample = HextreeWeightedPoolXYZ()
        self.proj = MLP(channels[-2], 2*dim, channels[-1])

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.mlps[i](data)
            data = self.downsample(data, hextree, depth_i)
            
        data = self.proj(self.norm(data))
        return data
    

class PatchEmbed2(torch.nn.Module):

    def __init__(self, patch_size:int = 32, in_dim: int = 4, dim: int = 96, num_stages: int = 2, nempty: bool = True, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.channels = [in_dim] + [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.mlps = torch.nn.ModuleList(
            [MLP(self.patch_size * self.channels[i], 
                 self.patch_size * self.channels[i+1], 
                 self.patch_size * self.channels[i+1]) for i in range(self.num_stages)])
        self.norm = torch.nn.LayerNorm(self.channels[-2])
        self.downsample = HextreeAvgPoolXYZ()
        self.proj = MLP(self.channels[-2], 2*dim, self.channels[-1])

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        for i in range(self.num_stages):
            nnum_t = data.shape[0]
            num = self.patch_size - nnum_t % self.patch_size
            tail = data.new_full((num,) + data.shape[1:], 0)
            data = torch.cat([data, tail], dim=0)
            data = data.view(-1, self.patch_size * self.channels[i])
            
            data = self.mlps[i](data)
            data = data.view(-1, self.channels[i+1])
            data = data[:nnum_t]
            
            depth_i = depth - i
            data = self.downsample(data, hextree, depth_i)
            
        data = self.proj(self.norm(data))
        return data
    

class PatchEmbed3(torch.nn.Module):

    def __init__(self, patch_size:int = 32, in_dim: int = 4, dim: int = 96, num_stages: int = 2, nempty: bool = True, init_depth: int = 9, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.channels = [in_dim] + [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.mlp = MLP(self.patch_size * self.channels[0], 
                 self.patch_size * self.channels[1], 
                 self.patch_size * self.channels[1])
        self.norm = torch.nn.LayerNorm(self.channels[-1])
        self.downsample = torch.nn.ModuleList([
            HextreeWeightedPoolXYZ(self.channels[i+1], self.channels[i+2], init_depth-i, init_depth-i-1)
            for i in range(self.num_stages)])
        self.proj = MLP(self.channels[-1], 2*dim, self.channels[-1])

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        
        nnum_t = data.shape[0]
        num = self.patch_size - nnum_t % self.patch_size
        tail = data.new_full((num,) + data.shape[1:], 0)
        data = torch.cat([data, tail], dim=0)
        data = data.view(-1, self.patch_size * self.channels[0])
        
        data = self.mlp(data)
        data = data.view(-1, self.channels[1])
        data = data[:nnum_t]
        
        for i in range(self.num_stages):            
            data = self.downsample[i](data, hextree)
            
        data = self.proj(self.norm(data))
        return data
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    # cosine time embedding
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, t):
        """
        t: (batch_size,) or (batch_size, 1)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        half_dim = self.embed_dim // 2
        device = t.device

        emb_scale = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb_freq = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        emb = t[:, None] * emb_freq[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        emb = self.linear(emb)
        return emb

class FlowModelBase(nn.Module):
    def __init__(self):
        super(FlowModelBase, self).__init__()
    def forward(self, x, t):
        raise NotImplementedError("FlowModelBase is an abstract class. Please implement the forward method.")

class SimpleFlowModel(FlowModelBase):
    def __init__(self,
                 input_dim=2,
                 time_dim=8,
                 hidden_dim=32):
        
        super(SimpleFlowModel, self).__init__()
        self.time_embedding = TimeEmbedding(embed_dim=time_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        """
        Args:
            x: (batch_size, input_dim) 
            t: (batch_size,) or (batch_size, 1)
        Returns:
            v: (batch_size, input_dim)  flow velocity
        """
        t_emb = self.collect_time_embeddings(t)
        h = torch.cat([x, t_emb], dim=1)
        v = self.net(h)
        return v

    def collect_time_embeddings(self, t):
        return self.time_embedding(t)

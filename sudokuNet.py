import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
        # Time embedding을 채널에 더해주는 레이어
        self.mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        # Convolution
        h = self.relu(self.conv1(x))
        # Time Embedding Injection
        time_emb = self.relu(self.mlp(t))
        # (Batch, Channel) -> (Batch, Channel, 1, 1)로 맞춰서 더하기
        h = h + time_emb[(..., ) + (None, ) * 2]
        
        h = self.relu(self.bn(self.conv2(h)))
        return h

class SudokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 9 # One-hot 채널
        hidden_dim = 128
        time_dim = 32
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial Conv
        self.inc = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        
        # ResNet Blocks (Pooling 없이 깊게만 쌓음)
        self.b1 = Block(hidden_dim, hidden_dim, time_dim)
        self.b2 = Block(hidden_dim, hidden_dim, time_dim)
        self.b3 = Block(hidden_dim, hidden_dim, time_dim)
        self.b4 = Block(hidden_dim, hidden_dim, time_dim)
        self.b5 = Block(hidden_dim, hidden_dim, time_dim)
        self.b6 = Block(hidden_dim, hidden_dim, time_dim)
        
        # Output Conv (다시 9채널로 복원)
        self.outc = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

    def forward(self, x, t):
        # t가 (Batch,) 형태의 정수 텐서로 들어옴
        t = self.time_mlp(t)
        
        x = self.inc(x)
        x = self.b1(x, t) + x # Residual Connection
        x = self.b2(x, t) + x
        x = self.b3(x, t) + x
        x = self.b4(x, t) + x 
        x = self.b5(x, t) + x
        x = self.b6(x, t) + x
        
        output = self.outc(x)
        return output
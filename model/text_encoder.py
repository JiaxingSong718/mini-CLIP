import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10,embedding_dim=16)
        self.dense1 = nn.Linear(in_features=16,out_features=64)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features=64,out_features=16)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(in_features=16,out_features=8)
        self.norm = nn.LayerNorm(normalized_shape=8)

    def forward(self,x):
        x = self.embedding(x)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.linear(x)
        x = self.norm(x)
        return x
    
if __name__ == '__main__':
    x = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    text_encoder = TextEncoder()
    print(text_encoder(x).shape)
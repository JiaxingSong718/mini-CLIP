import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2,padding=0)

        self.relu2 = nn.ReLU()

    def forward(self,x):
        Z = self.relu1(self.norm1(self.conv1(x)))
        Z = self.norm2(self.conv2(Z))
        res = self.conv3(x)
        return self.relu2(res+Z)
    
class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual1 = ResidualBlock(in_channels=3,out_channels=16,stride=2)
        self.residual2 = ResidualBlock(in_channels=16,out_channels=4,stride=2)
        self.residual3 = ResidualBlock(in_channels=4,out_channels=1,stride=2)
        self.linear = nn.Linear(16,10)
        self.norm = nn.LayerNorm(10)

    def forward(self,x):
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.linear(x.view(x.size(0),-1))
        x = self.norm(x)
        return x
    
if __name__ == '__main__':
    x = torch.randn(1,1,28,28)
    img_enc = ImageEncoder()
    print(img_enc(x).shape)


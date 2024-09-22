import torch
from torch import nn
from model.image_encoder import ImageEncoder
# from text_encoder import TextEncoder
from model.text_encoder_MHA import TextEncoder

class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(embedding_size=64,vocab_size=14,q_k_size=64,v_size=128,f_size=256,head=3,nblocks=3,class_num=10,seq_max_len=20)

    def forward(self,img_x,text_x):
        img_embedding = self.img_encoder(img_x)
        text_embedding = self.text_encoder(text_x)
        return img_embedding @ text_embedding.T
    
if __name__ =='__main__':
    clip = CLIP()
    img_x = torch.randn(10,1,28,28)
    text_x = torch.tensor([[ 0,  3,  2,  1, 13],
        [ 0,  3,  2,  1,  8],
        [ 0,  3,  2,  1,  6],
        [ 0,  3,  2,  1, 11],
        [ 0,  3,  2,  1,  7],
        [ 0,  3,  2,  1,  5],
        [ 0,  3,  2,  1,  4],
        [ 0,  3,  2,  1,  9],
        [ 0,  3,  2,  1, 10],
        [ 0,  3,  2,  1, 12]], dtype=torch.long)
    y = clip(img_x,text_x)
    print(y)
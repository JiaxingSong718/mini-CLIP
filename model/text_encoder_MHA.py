import torch
from torch import nn
from model.Embedding_and_Position import EmbeddingwithPosition
from model.Encoder_Block import EncoderBlock

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, q_k_size, v_size, f_size, head, nblocks, class_num, dropout=0.1, seq_max_len=5000) -> None:
        super().__init__()
        self.class_head = nn.Parameter(torch.rand(1,1,embedding_size))
        self.embedding = EmbeddingwithPosition(vocab_size=vocab_size, embedding_size=embedding_size)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.encoder_blocks.append(EncoderBlock(embedding_size=embedding_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size, head=head))
        
        self.class_Linear = nn.Linear(embedding_size,class_num)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self,x): #x:(batch_size,seq_len)
        x = self.embedding(x)
        class_head = self.class_head.expand(x.size(0),1,x.size(2)) # class_head:(batch_size,1,embedding_size)
        x = torch.cat((class_head, x), dim=1)
        for block in self.encoder_blocks:
            x = block(x)#x:(batch_size,seq_len,embedding_size)
        output = self.class_Linear(x[:,0,:])
        return self.softmax(output)
    
if __name__ == '__main__':
    
    batch = torch.tensor([[ 0,  3,  2,  1, 13],
        [ 0,  3,  2,  1,  8],
        [ 0,  3,  2,  1,  6],
        [ 0,  3,  2,  1, 11],
        [ 0,  3,  2,  1,  7],
        [ 0,  3,  2,  1,  5],
        [ 0,  3,  2,  1,  4],
        [ 0,  3,  2,  1,  9],
        [ 0,  3,  2,  1, 10],
        [ 0,  3,  2,  1, 12]], dtype=torch.long)
    print(batch.size())

    encoder = TextEncoder(embedding_size=128,vocab_size=14,q_k_size=256,v_size=512,f_size=512,head=5,nblocks=3,class_num=10,seq_max_len=100)
    z = encoder(batch)
    print(z)



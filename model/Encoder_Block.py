from torch import nn
import torch
from model.MultiHeadAttention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, q_k_size, v_size, f_size, head) -> None:
        super().__init__()
        self.Multihead_attention = MultiHeadAttention(embedding_size, q_k_size, v_size, head) #多头注意力
        self.z_linear = nn.Linear(head * v_size, embedding_size) #将多头注意力的输出调整为embedding_size,方便后面的残差连接相加
        self.addnorm1 = nn.LayerNorm(embedding_size) #按照last dim做norm

        #Feed Back
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, embedding_size)
        )

        self.addnorm2 = nn.LayerNorm(embedding_size) #按照last dim做norm

    def forward(self, x): #x:(batch_size,seq_len,embedding_size)
        z = self.Multihead_attention(x, x) #x: (batch_size, seq_len, head*v_size)
        z = self.z_linear(z) #z:(batch,seq_len,embedding_size)
        output1 = self.addnorm1(z + x) #output:(batch,seq_len,embedding_size)

        output2 = self.feedforward(output1) #output:(batch,seq_len,embedding_size)
        EncoderBlock_output = self.addnorm2(output2 + output1) #output:(batch,seq_len,embedding_size)
        return EncoderBlock_output

# if __name__ == '__main__':
#     embedding = EmbeddingwithPosition(len(de_vocab),128)
#     de_tokens, de_ids = de_preprocess(train_dataset[0][0])
#     de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)

#     embedding_result = embedding(de_ids_tensor.unsqueeze(0))
#     attention_mask = torch.zeros((1, de_ids_tensor.size()[0], de_ids_tensor.size()[0]))
    
#     # 5个encoder block堆叠
#     encoder = []
#     for i in range(5):
#         encoder.append(EncoderBlock(embedding_size=128, q_k_size=256, v_size=512, f_size=512, head=8))
    
#     encoder_output = embedding_result
#     for i in range(5):
#         encoder_output = encoder[i](encoder_output, attention_mask)

#     print('encoder_output:', encoder_output.size())



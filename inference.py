from dataset.dataset import CIFAR10, en_preprocess
from model.CLIP import CLIP
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

class_label = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = CIFAR10()

model = CLIP().to(DEVICE)

model.load_state_dict(torch.load('./checkpoints/CLIP_model_MHA2.pth'))

model.eval()

"""
1.对图片进行分类
"""
image,label = dataset[22]
print('正确分类：',label)
# plt.imshow(image.permute(1,2,0))
# plt.show()
targets = torch.arange(0,10).to(DEVICE)
print(targets)
label = torch.tensor([en_preprocess(f"A photo of a {class_label[l.item()]}") for l in targets]).to(DEVICE)
print(label)
output = model(image.unsqueeze(0).to(DEVICE),label.to(DEVICE))
print(output)
print('CLIP分类：','A photo of a '+ class_label[output.argmax(-1).item()])

"""
2.图片相似度
"""
# other_images = []
# other_labels = []
# for i in range(101):
#     other_image,other_label = dataset[i]
#     other_images.append(other_image)
#     other_labels.append(other_label)

# # 其他100张图片向量
# other_images_embedding = model.img_encoder(torch.stack(other_images,dim=0).to(DEVICE))

# image_embedding = model.img_encoder(image.unsqueeze(0).to(DEVICE))

# # 计算当前图片与其他100张图片的相似度
# logtis = image_embedding @ other_images_embedding.T
# value,indexs = logtis[0].topk(5)
# plt.figure(figsize=(10,10))
# for i,img_index in enumerate(indexs):
#     plt.subplot(1,5,i+1)
#     plt.imshow(other_images[img_index].permute(1,2,0))
#     plt.title(other_labels[img_index])
#     plt.axis('off')
# plt.show()
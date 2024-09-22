from dataset.dataset import CIFAR10, en_preprocess
from model.CLIP import CLIP
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class_label = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = CIFAR10()

model = CLIP().to(DEVICE)

try:
    model.load_state_dict(torch.load('./checkpoints/CLIP_model_MHA2.pth'))
except:
    pass

optimzer = torch.optim.Adam(model.parameters(),lr=0.00001)
loss_fn = nn.CrossEntropyLoss()

epochs = 400000
BATCH_SIZE = 64
TARGET_COUNT = 10
best_loss = float('inf')

dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

for epoch in range(epochs):
    while True:
        imgs,labels=next(iter(dataloader))
        if torch.unique(labels).shape[0]<TARGET_COUNT:  # 未覆盖10种数字
            continue
        # 挑选出10个数字
        target=set()    
        indexes=[]
        for j in range(BATCH_SIZE):
            if labels[j].item() in target:
                continue 
            target.add(labels[j].item())
            indexes.append(j)
            if len(target)==TARGET_COUNT:
                break
        imgs=imgs[indexes]
        labels=labels[indexes]
        label = torch.tensor([en_preprocess(f"A photo of a {class_label[l.item()]}") for l in labels])
        break
    # print(epoch)
    # print(labels)
    Z = model(imgs.to(DEVICE),label.to(DEVICE))
    targets = torch.arange(0,TARGET_COUNT).to(DEVICE)
    # print(targets)
    loss_i = loss_fn(Z,targets)
    loss_j = loss_fn(Z.permute(1,0),targets)
    loss = (loss_i + loss_j)/2

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if epoch % 100 == 0:
        print('iter:{}, loss:{}'.format(epoch,loss))
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), './checkpoints/CLIP_model_MHA2.pth')
        print('Model saved with loss:', best_loss)
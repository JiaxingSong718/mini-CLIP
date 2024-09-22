from torch.utils.data import Dataset
from torchvision import transforms
from torchtext.data.utils import get_tokenizer
import torchvision
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

class_label = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
class CIFAR10(Dataset):
    def __init__(self,is_train=True) -> None:
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10('./dataset/data',train=is_train,download=True)
        self.img_convert = transforms.Compose([
            transforms.PILToTensor()
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        img,label = self.dataset[index]
        return self.img_convert(img)/255.0, label
    
# 'A photo of a ' + class_label[label]
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
dataset = CIFAR10()
# print('A photo of a ' + class_label[dataset[0][-1]]); exit()

tokens = [] # token列表
for en in tqdm(dataset, desc="Processing"):
    tokens.append(tokenizer('A photo of a ' + class_label[en[-1]]))


en_vocab = build_vocab_from_iterator(tokens) # 英语token词表

# 句子特征预处理

def en_preprocess(en_sentence):
    tokens = tokenizer(en_sentence)
    ids = en_vocab(tokens)
    return ids
    
if __name__ == '__main__':
     # 词表大小
    print('en vocab:', len(en_vocab))

    # 特征预处理
    en_sentence=dataset[5][-1]
    print('en preprocess:',*en_preprocess(en_sentence))
    import matplotlib.pyplot as plt
    dataset = CIFAR10()
    img,label = dataset[0]
    print(label)
    plt.imshow(img.permute(1,2,0))
    plt.show()
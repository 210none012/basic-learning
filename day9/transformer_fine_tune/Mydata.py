from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset

class Mydataset(Dataset):
    #初始化数据
    def __init__(self,spilit):
        self.dataset = load_dataset("lansinuote/ChnSentiCorp")
        if spilit == "train":
            self.dataset = self.dataset["train"]
        elif spilit == "validation":
            self.dataset = self.dataset["validation"]
        elif spilit == "test":
            self.dataset = self.dataset["test"]
        else:
            print("数据集名称错误！")

    #获取数据长度
    def __len__(self):
        return len(self.dataset)
    #对数据定制化处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label =self.dataset[item]["label"]
        return text, label

if __name__ == '__main__':
    dataset = Mydataset("validation")
    for data in dataset:
        print(data)
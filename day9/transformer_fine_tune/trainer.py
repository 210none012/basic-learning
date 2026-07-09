import torch
from Mydata import Mydataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from torch.optim import AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100

token = BertTokenizer.from_pretrained("bert-base-chinese")
#自定义函数，对数据进行编码处理
def collate_fn(data):
    sentes = [i[0] for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,    #要编码的文本
        truncation=True,
        padding="max_length",               #填充至max_length
        max_length=500,
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids,labels
#创建数据集
train_dataset = Mydataset("train")
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True,
    drop_last = True,
    collate_fn=collate_fn
)

#开始训练
if __name__ == '__main__':
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(),lr=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            #将数据放到DEVICE上
            input_ids,attention_mask,token_type_ids,labels = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(out,labels)

            optimizer.zero_grad()   #清空上一批次的梯度，防止累计
            loss.backward()         #计算当前梯度
            optimizer.step()        #根据梯度优化

            if i%5 == 0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item()/len(labels)
                print(epoch,i,loss.item(),acc)

            #保存模型参数
            torch.save(model.state_dict(),f"params/{epoch}bert.pt")
            print(epoch,"参数保存成功")
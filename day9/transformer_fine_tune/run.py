import torch
from net import Model
from transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = ["负向评价","正向评价"]
print(DEVICE)
model = Model().to(DEVICE)

token = BertTokenizer.from_pretrained("bert-base-chinese")

def collate_fn(data):
    #从控制台输入获取
    sentes = []
    sentes.append(data)
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,
        truncation=True,
        padding="max_length",
        max_length=500,
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]               #输入token的ID张量，句子的每一token和padding的组合
    attention_mask = data["attention_mask"]     #注意力掩码张量，分离无用的padding填充部分
    token_type_ids = data["token_type_ids"]     #提取 token 类型 ID 张量，用于区分不同句子

    return input_ids, attention_mask, token_type_ids

def test():
    model.load_state_dict(torch.load("params/13bert.pt"))
    model.eval()    #评估模式
    while True:
        data = input("请输入测试数据（输入q退出）：")
        if data == "q":
            print("测试结束")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids,attention_mask,token_type_ids = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids,attention_mask,token_type_ids)
            out = out.argmax(dim=1)     #在类别维度上取最大值索引
            print("模型判断：",names[out],"\n")

if __name__ == '__main__':
    test()
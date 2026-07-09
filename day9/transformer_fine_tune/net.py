from transformers import BertModel
import torch

#定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)

print(pretrained)

#定义下游任务模型
class Model(torch.nn.Module):
    #模型结构设计
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768,2)    #BERT的最后一层输出为768

    def forward(self,input_ids,attention_mask,token_type_ids):
        #上游任务不参与训练
        with torch.no_grad():   #只参与前向传播，不参与反向训练
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #下游任务参与训练
        out = self.fc(out.last_hidden_state[:,0])
        out = out.softmax(dim=1)
        return out
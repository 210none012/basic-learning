import torch
from torch import optim, nn

#Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank    #低阶矩阵的秩
        self.A = nn.Linear(in_features, rank, bias=False)   #矩阵A（降维）
        self.B = nn.Linear(rank, out_features, bias=False)  #矩阵B（升维）
        self.A.weight.data.normal_(mean=0.0, std=0.02)      #高斯初始化
        self.B.weight.data.zero_()                          #全0初始化

    def forward(self, x):
        return self.B(self.A(x))    #前向传播

def apply_lora(model, rank=16):
    #遍历model的子模块，对输入输出维度相同的线性层（方阵）进行LoRA
    for name, module in model.name_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = Lora(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)   #动态为module对象添加LoRA实例
            original_forward = module.forward   #引用对象

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forwward_with_lora #输出为原始输出+Lora旁路输出

def load_lora(model, path):
    #从路径加载并映射到设备
    state_dict = torch.load(path, map_location=model.device)
    #带有module.说明被nn.parallel包装过，去掉前缀，使key与当前命名匹配
    state_dict = {(k[7:] if k.startwith('module.') else k): v for k,v in state_dict.items()}
    for name, module in model.name_modules():
        if hasattr(module, 'lora'): #判断模块是否有lora属性
            #去掉前缀
            lora_state = {k.replace(f'{name}.lora.',''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.lora_state_dict(lora_state)

def save_lora(model, path): 
    raw_model = getattr(model, '_orig_mod', model)  #获取原始模型
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            #v转为half精度节省空间，并转移到cpu上
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k,v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    #保存为单独的权重文件
    torch.save(state_dict, path)

def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k:v.cpu().half() for k,v in raw_model.state_dict().items()}
    for name, module in raw_model.named_modules():
        #获取不含lora的key
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                #若挂载了lora则将lora的值加到原始权重上
                #@表示矩阵乘法
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
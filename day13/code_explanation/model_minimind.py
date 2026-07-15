import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activation import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

#########################Config#########################
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super.__init()__(**kwargs)
        self.hidden_size = hidden_size  #隐藏层维度
        self.num_hidden_layers = num_hidden_layers  #层数
        self.use_moe = use_moe  #启用多专家
        self.dropout = kwargs.get("dropout", 0.0)   #dropout率
        self.vocab_size = kwargs.get("vocab_size", 6400)    #词表大小
        self.bos_token_id = kwargs.get("bos_token_id", 1)   #设置BOS（Begin of Sequence)标记的token ID，表示序列开始
        self.eos_token_id = kwargs.get("eos_token_id", 2)   #设置EOS（End of Sequence)标记的token ID，表示序列结束
        self.flash_attn = kwargs.get("flash_attn", True)    #是否使用flash_attention加速
        self.num_attention_heads = kwargs.get("num_attention_heads", 8) #注意力头
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4) #KV头（<注意力头）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)    #每个头的维度=总维度/注意力头数）
        self.hidden_act = kwargs.get("hidden_act", 'silu')  #激活函数
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)    #FFN中间层维度
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768) #最大序列长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)    #RMS的防除0系数
        self.rope_theta = kwargs.get("rope_theta", 1e6)         #RoPE基数
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  #输入输出词嵌入是否共享权重
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)   #是否启用YaRN
        self.rope_scaling = {
            "beta_fast": 32,   #高频阈值，大于此值不缩放
            "beta_slow": 1,     #低频阈值，小于此值不缩放
            "factor": 16,       #扩张倍数
            "original_max_position_embeddings": 2048,   #模型训练时的最大序列长度
            "attention_factor": 1.0,    #注意力分数缩放因子
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4) #专家数量
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1) #每个token选几个专家
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)    #路由辅助损失系数，用于平衡专家负载
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)    #路由辅助损失的权重系数，用于平衡专家负载

#######################Model#########################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) #可学习的缩放参数

    #RMSNorm只做缩放，不做平移
    def norm(self, x):
        #每个元素平方求平均的开方的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) +self.eps)

    def forward(self, x):
        #先用float计算保证稳定性，再转回x类型
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    #先按步长为2,取dim，再保留前dim/2个元素，保证只有dim/2个数
    freqs, attn_factor = 1.0/(rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float()/dim)), 1.0   #freq=1/1000000^(2i/d),(i=1,2...31)
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("sttention_factor", 1.0)
        )
        #请求序列end长度大于训练长度orig_max时缩放，避免大于推理长度
        if end/orig_max > 1.0:
            #解出b在哪一维度（i)
            inv_dim = lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
            #定义缩放边界
            low,high = max(math.floor(inv_dim(beta_fast)),0), min(math.ceil(inv_dim(beta_slow)), dim//2-1)
            #以low为起点，low以前为0，到high之间成线性，high以后为1
            #torch.clamp(n,min,max):保证所有值在min与max之间
            ramp = torch.clamp((torch.arange(dim//2, device=freqs.device).float()-low)/max(high-low, 0.001), 0, 1)
            freqs = freqs*(1-ramp+ramp/factor)  #缩放
    t = torch.arange(end, device=freqs.device)  #生成位置索引
    freqs = torch.outer(t, freqs).float()       #torch.outer()计算外积，位置×频率
    #拼接补全dim维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)*attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)*attn_factor
    return freqs_cos, freqs_sin

def apppl_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    #将后半部分取反与前半部分拼接
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1]//2:],x[..., :x.shape[-1]//2]),dim=-1)
    #隔一个数为一组（0,2）（1,3）
    q_embed = ((q*cos.unsqueeze(unsqueeze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k*cos.unsqueeze(unsqueeze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

#扩展x的kv头数
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    #bs-batch size,slen-序列长度，num_key_value_heads-KV头数量，head_dim-每个头的维度
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_attention_heads
        self.n_rep = self.n_local_heads//self.n_local_kv_heads
        self.head_dim - config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size,config.num_attention_heads*self.head_dim,bias=False)
        self.k_proj = nn.Linear(config.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj = nn.Linear(config.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        #将多头拼接后的信息融合成最终输出
        self.o_proj = nn.Linear(config.num_attention_heads*self.head_dim,config.hidden_size,bias=False)
        self.q_norm = RMSNoem(self.head_dim,eps=config.rms_norm_eps)
        self.k_norm = RMSNoem(self.head_dim,eps=config.rms_norm_eps)
        self.attn_droupout = nn.Droupout(config.droupout)
        self.resid_droupout = nn.Droupout(config.droupout)
        self.droupout = config.droupout
        #同时满足pytorch版本与启用
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention') and config.flash_attn

def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    #对于不同数据，shape不同维度表示意义不同
    bsz, seq_len, _ =x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    #tensor.view(*shape)用于重塑形状，类似于numpy的reshape
    xq = xq.view(bsz, seq, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seq, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq, self.n_local_kv_heads, self.head_dim)
    xq, xk = self.q_norm(xq), self.k_norm(xk)
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    #拼接历史缓存，在推理时，token需要逐个生成
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if(use_cache) else None
    #transpose为转置矩阵；repeat_kv将kv头数扩展至与q头相同
    xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1,2))
    #条件：pytorch支持且启用，序列长度大于1，is_causal表示是否为因果注意力
    if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
        #flash attention分支
        #自动计算缩放点积注意力（可以自动选择并调用优化）
        output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
    else:
        #标准注意力分支
        scores = (xq @ xk.transpose(-2, -1))/math.sqrt(self.head_dim)   #计算注意力分数
        #torch.full():创建一个seq_len*seq_len的全-inf矩阵
        #因果掩码时，对最后seq_len列添加掩码;triu(1)生成上三角矩阵，保留对角线以上元素，其余置0
        #scores的形状为[batch,heads,query_len,key_len]，query_len为当前序列长度，key_len为总长度
        if self.is_causal:scores[:,:,:,-seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)   
        #Padding掩码器
        #unsqueeze（）在对应位置插入1个维度为1的维度，使attention_mask与scores匹配
        #1表示有效token，0表示掩码，乘上-1e9，padding变为极大的负数
        if attention_mask is not None: scores += (1.0-attention_mask.unsqueeze(1).unsqueeze(2))*-1e9
        #softmax将scores转化为和为1的概率分布,dim=-1即对每一query，key的和为1
        #reshape要求总元素个数不变，-1代表位置元素个数为scores的后两个维度相乘
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
    output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
    output = self.resid_dropout(self.o_proj(output))
    return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size,bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size,bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    #SwiGLU
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        #创建多个专家层
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        #铺平，方便进行一些对二维的操作
        x_flat = x.view(-1, hidden_dim)
        #计算每个专家的概率分数
        scores = F.softmax(self.gate(x_flat), dim=-1)
        #scores=[num_tokens, num_experts];k:每个token选择的专家数，dim=-1：在专家维度取Topk
        #topk_weight为最高k位的权重
        #top_idx为最高k位的索引
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_top, dim=-1, sorted=False)
        #归一化
        if self.config.norm_topk_prob: topk_weight/(topk_weight.sum(dim=-1, keepdim=True)+1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            #找出分配给当前专家的token，并转化为布尔张量
            mask = (topk_idx == i)
            if mask.any():
                #标记那些行选择了该专家
                #nonzero()获取非0行的索引
                token_idx = mask.any(dim=-1).nonzero().flatten()
                #提取对应token的权重，并转化为列向量
                weight = topk_weight[mask].view(-1,1)
                #对应token经专家权重计算并乘以对应专家权重累加输出到张量y的对应位置
                y.index_add_(0, token_idx, (expert(x_flat[token_idx])*weight).to(y.dtype))
            elif self.training:
                #如果有专家没有被选中
                y[0, 0] += 0*sum(p.sum() for p in expert.parameters())
        #训练时且损失系数大于0
        if self.training and self.config.router_aux_loss_coef > 0:
            #转换为one-hot编码后求每一专家的平均值
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            #每一专家被选中的概率乘以对应分数，然后求和，再乘以专家数量和辅助损失系数
            self.aux_loss = (load*scores.mean(0)).sum()*self.config.num_experts*self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, 
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        #注册不参与梯度更新参数，persistent=True时，参数保存在字典中
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        #past_key_value可能是BynamicCache对象（有.layers属性），此时设置为None，走标准缓存路径
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        #设置past_key_value的起始位置
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        #数据丢失时需要重新计算RoPE
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        #所有层的KV缓存
        presents = []
        #zip：将2个表按索引配对，同时遍历
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

#PreTrainedModel:提供模型加载、保存
#GenerationMixin：提供自回归生成能力
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    #指定配置类用于from_pretrained自动加载配置
    config_class = MiniMindConfig
    #权重共享声明，当保存或加载模型时，会检查该字典，第一个参数为当前模型中的参数名，第二个参数为共享权重来源参数名
    #保存时：lm_head.weight 不会单独保存，而是从model.embed_tokens.weight复制
    #加载时：lm_head.weight 会被设置为 model.embed_tokens.weight 的引用
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        #输出投影层，从输出映射到词表大小，预测概率
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        #权重共享
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head_weight
        #后初始化，防止重写初始化，导致一些项目未初始化成功
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        #取最后logits_to_keep位的token切片
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:,slice_indices,:])
        #推理时loss保持为None
        loss = None
        #labels非None时表示在训练
        if labels is not None:
            #去掉预测的最后一个token，以及label的第一个token
            #符合实际中用前面的已知量预测下一token
            #contiguous用于确保张量是连续的，在不连续时，重新排序使其连续
            x, y = logits[..., :-1, :].contiguous(),labels[..., 1:].contiguous()
            #交叉熵损失，ignore_index=-100：忽略标签为-100的位置
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1),ignore_index=-100)
        #统一返回各种输出结果
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    #禁用梯度计算和自动求导，用于推理模式
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        #从kwargs取出input_ids,没有则使用inputs，并复制多份
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        #如果有流式输出器则将input_ids推送给流
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            #获取当前缓存长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            #之传入新增的token
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            #只取最后一个token的预测，除以temperature进行缩放
            logits = outputs.logits[:, -1, :] / temperature
            #惩罚重复出现的token
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    #记录已出现的token及其分数，降低其分数
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            #multinomial:按概率分布采样（随机）
            #argmax：去概率最高的token（贪心）
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            #next_token.new_full创建[batch,1]形状的张量，填充eos，强制终止生成
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                #比较是否最后一位均为eos结束符
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
    
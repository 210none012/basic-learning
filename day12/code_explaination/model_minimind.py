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

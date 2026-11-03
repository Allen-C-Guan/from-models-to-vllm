"""小型post-LN编码器解码器；用于机制验证，不是WMT复现。"""
import math
import torch
from torch import nn

def positions(n,d,dtype,device):
    p=torch.arange(n,dtype=dtype,device=device)[:,None]
    omega=10000**(-torch.arange(0,d,2,dtype=dtype,device=device)/d)
    out=torch.empty(n,d,dtype=dtype,device=device)
    out[:,0::2]=torch.sin(p*omega);out[:,1::2]=torch.cos(p*omega)
    return out

class Attention(nn.Module):
    def __init__(self,d,h):
        super().__init__();assert d%h==0
        self.h=h;self.dk=d//h
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
        self.last_weights=None
    def forward(self,x,memory,allowed):
        # x:(B,m,d), memory:(B,n,d), allowed可广播到(B,h,m,n)，True允许。
        b,m,d=x.shape;n=memory.shape[1]
        split=lambda t:t.reshape(b,-1,self.h,self.dk).transpose(1,2)
        q,k,v=split(self.q(x)),split(self.k(memory)),split(self.v(memory))
        scores=q@k.transpose(-1,-2)/math.sqrt(self.dk)
        if not allowed.any(-1).all():raise ValueError('有效查询不允许全遮罩')
        w=torch.softmax(scores.masked_fill(~allowed,float('-inf')),dim=-1)
        self.last_weights=w.detach()
        z=(w@v).transpose(1,2).reshape(b,m,d)
        return self.out(z)

class Block(nn.Module):
    def __init__(self,d,h,ff,drop,decoder):
        super().__init__();self.self_attn=Attention(d,h);self.decoder=decoder
        if decoder:self.cross=Attention(d,h)
        self.ff=nn.Sequential(nn.Linear(d,ff),nn.ReLU(),nn.Linear(ff,d))
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(3 if decoder else 2)])
        self.drop=nn.Dropout(drop)
    def forward(self,x,allowed,memory=None,source_allowed=None):
        x=self.norms[0](x+self.drop(self.self_attn(x,x,allowed)))
        if self.decoder:
            x=self.norms[1](x+self.drop(self.cross(x,memory,source_allowed)))
        return self.norms[-1](x+self.drop(self.ff(x)))

class Translator(nn.Module):
    def __init__(self,vocab=9,d=32,h=4,layers=2,ff=64,drop=.1,use_positions=True):
        super().__init__();self.d=d;self.use_positions=use_positions
        self.source=nn.Embedding(vocab,d);self.target=nn.Embedding(vocab,d)
        self.encoder=nn.ModuleList([Block(d,h,ff,drop,False) for _ in range(layers)])
        self.decoder=nn.ModuleList([Block(d,h,ff,drop,True) for _ in range(layers)])
        self.drop=nn.Dropout(drop)
    def embed(self,ids,table):
        x=table(ids)*math.sqrt(self.d)
        if self.use_positions:x=x+positions(ids.shape[1],self.d,x.dtype,x.device)
        return self.drop(x)
    def encode(self,source):
        allowed=(source!=0)[:,None,None,:]
        x=self.embed(source,self.source)
        for block in self.encoder:x=block(x,allowed)
        return x,allowed
    def decode(self,target,memory,source_allowed):
        m=target.shape[1]
        causal=torch.ones(m,m,dtype=torch.bool,device=target.device).tril()
        allowed=causal[None,None,:,:] & (target!=0)[:,None,None,:]
        x=self.embed(target,self.target)
        for block in self.decoder:x=block(x,allowed,memory,source_allowed)
        return x@self.target.weight.T  # 目标输入/输出绑定；不含额外最终LN。
    def forward(self,source,target):
        return self.decode(target,*self.encode(source))
    @torch.no_grad()
    def generate(self,source,max_steps=4):
        memory,allowed=self.encode(source)
        out=torch.ones(source.shape[0],1,dtype=torch.long,device=source.device)
        done=torch.zeros(source.shape[0],dtype=torch.bool,device=source.device)
        for _ in range(max_steps):
            nxt=self.decode(out,memory,allowed)[:,-1].argmax(-1)
            nxt=torch.where(done,torch.full_like(nxt,2),nxt);done|=nxt==2
            out=torch.cat([out,nxt[:,None]],dim=1)
        return out[:,1:]

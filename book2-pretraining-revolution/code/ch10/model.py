# model.py —— Book2 ch10 大项目 I 定版模型：从零实现的 GPT-2 124M（三插槽 Block，Book3 改装契约）
# 所属章节：Book2 第 10 章 §10.2-10.3（drafts/Book2-预训练革命/ch10-大项目I.md；ch11官方权重对拍复用）
# 单轨依赖链：ch04/warmup_ablation.py 的参数化 GPT（前身）→ 本文件（正式定版）→ ch11 复用（官方权重对拍）。
# 结构对照 nanoGPT（MIT，2022-12，结构对照非照抄）与 HF transformers 5.18.0 modeling_gpt2.py（仅对照，命名贴近以便 ch11 键映射）。
# 运行方式（自测：参数 assert + 初始化警示实验）：python model.py            （CPU前向，不训练；--counts-only仅计数）
# 被引用：from model import GPT2, GPT2Config（train.py / shakespeare.py 同目录）
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 超参与 124M 口径 ----------------
# 官方口径（GPT-2 论文 §2.3 + openai-community/gpt2 config）：12 层 / 768 维 / 12 头 / ctx 1024 /
#   词表 50257 / 初始化 N(0,0.02) / gelu_new（GELU 的 tanh 近似）/ tie_word_embeddings=True。
# dropout：HF公开config三个pdrop=0.1；本书默认0.0，nanoGPT的shakespeare_char配置为0.2；
#   需要官方口径时 GPT2Config(dropout=0.1)。dropout 不影响参数量与权重对拍（eval 模式下关闭）。
GPT2_124M = dict(vocab_size=50257, n_positions=1024, n_layer=12, n_embd=768, n_head=12)


class GPT2Config:
    """GPT-2 配置（字段名对齐官方 config；n_positions 即 wpe 行数=上下文硬上限）。"""

    def __init__(self, vocab_size=50257, n_positions=1024, n_layer=12, n_embd=768,
                 n_head=12, dropout=0.0, initializer_range=0.02):
        sizes = (vocab_size, n_positions, n_layer, n_embd, n_head)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in sizes):
            raise ValueError("模型尺寸必须为正整数")
        if n_embd % n_head:
            raise ValueError("隐藏宽度须能被头数整除")
        if not 0 <= dropout < 1 or not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("dropout须在[0,1)，初始化标准差须为有限正数")
        self.vocab_size = vocab_size        # wte 行数（50257 = 256 字节 + 50000 合并 + 1 <|endoftext|>，见第 2 章）
        self.n_positions = n_positions      # wpe 行数（1024 = 学习式位置查表的硬上限，第 4 章手术③）
        self.n_layer = n_layer              # Block 个数
        self.n_embd = n_embd                # 残差流宽度 C
        self.n_head = n_head                # 注意力头数 h（head_dim = C/h = 64）
        self.dropout = dropout
        self.initializer_range = initializer_range


class CausalSelfAttention(nn.Module):
    """注意力插槽：合并式 QKV 投影 + 因果掩码 + 输出投影。输入 (B,n,C) -> 输出 (B,n,C)。

    c_attn 一次算出 Q|K|V 三段（GPT-2 官方 Conv1D(3C, C) 的同构实现，nn.Linear 排布）：
    权重加载时注意 HF Conv1D 是 (in,out) 排布、nn.Linear 是 (out,in)——ch11 键映射要转置。
    """

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        C = cfg.n_embd
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(C, 3 * C)                       # (B,n,C) -> (B,n,3C)
        self.c_proj = nn.Linear(C, C)                           # (B,n,C) -> (B,n,C)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # 因果掩码：下三角可见（第 i 个位置只准看 <= i）。persistent=False：不进 state_dict
        # HF各版本的掩码生成位置不同，加载时须核实际键清单，不按版本大类猜测。
        self.register_buffer("mask", torch.tril(torch.ones(cfg.n_positions, cfg.n_positions))
                             .view(1, 1, cfg.n_positions, cfg.n_positions), persistent=False)

    def forward(self, x):
        B, n, C = x.shape                                              # (B,n,C)
        hd = C // self.n_head                                          # 64
        q, k, v = self.c_attn(x).split(C, dim=2)                       # (B,n,3C) -> 各 (B,n,C)
        q = q.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
        k = k.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
        v = v.view(B, n, self.n_head, hd).transpose(1, 2)              # (B,h,n,hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                # (B,h,n,hd)·(B,h,hd,n) -> (B,h,n,n)
        att = att.masked_fill(self.mask[:, :, :n, :n] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))                   # (B,h,n,n)；softmax行和为1，训练时dropout后不必为1
        y = att @ v                                                    # (B,h,n,n)·(B,h,n,hd) -> (B,h,n,hd)
        y = y.transpose(1, 2).contiguous().view(B, n, C)               # (B,h,n,hd) -> (B,n,C)
        return self.resid_drop(self.c_proj(y))                         # (B,n,C)


class MLP(nn.Module):
    """前馈插槽：C -> 4C -> GELU(tanh 近似=gelu_new) -> C。输入 (B,n,C) -> 输出 (B,n,C)。

    4C=3072 沿袭原典比例；激活用 tanh 近似版 GELU（GPT-2 官方 config 的 gelu_new；
    Google原始BERT也使用tanh近似；库中同名函数仍应核对具体实现，见正文考据框）。
    """

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        C = cfg.n_embd
        self.c_fc = nn.Linear(C, 4 * C)                                # (B,n,C) -> (B,n,4C)
        self.c_proj = nn.Linear(4 * C, C)                              # (B,n,4C) -> (B,n,C)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """三插槽 pre-LN 残差块——Book3 改装契约的锚点。

    契约：块 = 两条残差支路，每条 = [归一化 ln_*] + [插槽子层 attn / mlp]；插槽内部怎么换
    （注意力变体、前馈变体、归一化变体）是 Book3 逐插槽改装的事，块级拓扑（pre-LN + 残差 +
    两支路次序）在本册定版。forward 输入/输出均为 (B,n,C)。

        x = x + attn(ln_1(x))    # 支路一：注意力插槽（LN 在入口 = pre-LN，第 4 章手术②）
        x = x + mlp(ln_2(x))     # 支路二：前馈插槽
    """

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)      # eps 默认 1e-5，与官方 GPT-2 一致（BERT 是 1e-12）
        self.attn = CausalSelfAttention(cfg)      # 插槽 1
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)                       # 插槽 2

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))           # (B,n,C) -> (B,n,C)
        x = x + self.mlp(self.ln_2(x))            # (B,n,C) -> (B,n,C)
        return x


class GPT2(nn.Module):
    """GPT-2 整机：wte + wpe + n_layer 个 Block + ln_f + tied lm_head。

    前向：idx (B,n) -> logits (B,n,V)；给 targets (B,n) 时返回全部目标位置的平均交叉熵标量（因果 LM 目标：
    用第 t 个位置的状态预测第 t+1 个 token——T5 的 span corruption 是同一预训练范式在
    encoder-decoder 侧的另一端，见第 5 章）。

    初始化 init="gpt2"（默认）：Linear/Embedding正态分布，标准差s=0.02、偏置清零，且两条残差支路的
    流出端投影（attn.c_proj / mlp.c_proj）重新以标准差s/sqrt(2*n_layer)初始化；这是本书复现约定，见正文原报告与代码的边界说明。
    init="default"：完全跳过自定义初始化（PyTorch 默认：Embedding ~ N(0,1)，Linear ~ kaiming 均匀），
    供「初始化警示」实验对照——此时 tied head 的 logit 方差约 C=768，初始 loss 会冲到数百而非
    ln(50257)≈10.82（实测见 model.py 自测与正文 10.2 节）。
    """

    def __init__(self, cfg: GPT2Config = None, init: str = "gpt2"):
        super().__init__()
        cfg = cfg or GPT2Config(**GPT2_124M)
        self.cfg = cfg
        self.block_size = cfg.n_positions
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # (V,C) 词嵌入（兼输出投影，tied）
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)    # (n_max,C) 学习式位置查表（手术③）
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)                    # 末尾补一层 LN（手术②的另一半）
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight                      # tied embeddings：输出投影=词嵌入（手术④）
        if init == "gpt2":
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():               # 残差流出端按深度缩放
                if pn.endswith("c_proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.n_layer))
        elif init != "default":
            raise ValueError(init)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def n_params(self, non_embedding: bool = False) -> int:
        """参数量。non_embedding=True 为 6ND 口径（扣 wte 与 wpe；tied head 与 wte 同存储天然不计）。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, n = idx.shape                                        # (B,n)
        if not 1 <= n <= self.block_size:
            raise ValueError(f"序列长须在1—{self.block_size}，收到{n}")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets必须与idx同为(B,n)；标签由调用方右移一位")
        pos = torch.arange(n, device=idx.device)                # (n,)
        x = self.drop(self.wte(idx) + self.wpe(pos))            # (B,n,C)+(n,C) -> (B,n,C)
        for blk in self.blocks:                                 # 12 × (B,n,C)
            x = blk(x)
        x = self.ln_f(x)                                        # (B,n,C)
        logits = self.head(x)                                   # (B,n,C)·(C,V) -> (B,n,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),   # (B*n,V)
                                  targets.reshape(-1))                    # (B*n,) -> 标量
        return logits, loss


# ---------------- 自测：计数与前向初始化诊断；不运行训练 ----------------
def initialization_report(batch=1, seq=64, seed=20261002):
    """CPU独立随机输入和标签，仅检查前向尺度；不是下一token训练批。"""
    if batch <= 0 or not 1 <= seq <= 1024:
        raise ValueError("批量须正，序列长须在1—1024")
    torch.manual_seed(seed)
    model = GPT2().eval()
    assert model.n_params() == 124_439_808
    assert model.n_params(True) == 85_056_000
    x = torch.randint(0, 50257, (batch, seq))
    y = torch.randint(0, 50257, (batch, seq))
    result = {"device": "cpu", "dtype": "float32", "torch": str(torch.__version__),
              "seed": seed, "batch": batch, "seq": seq,
              "label_protocol": "独立均匀随机标签；前向尺度诊断",
              "params_total": model.n_params(), "params_non_emb": model.n_params(True),
              "uniform_ce": math.log(50257), "small_scale_approx": math.log(50257)+768*.02**2/2}
    with torch.no_grad():
        logits, loss = model(x, y)
        result["gpt2_loss"] = loss.item()
        del logits, loss, model
        model = GPT2(init="default").eval()
        logits, loss = model(x, y)
        own = logits.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        result.update(default_loss=loss.item(), default_logits_std=logits.std().item(),
                      default_copy_rate=(logits.argmax(-1)==x).float().mean().item(),
                      default_own_logit_min=own.min().item(), default_own_logit_max=own.max().item())
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="参数对账及CPU初始化诊断，不运行训练")
    ap.add_argument("--counts-only", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path.cwd()/"log/book2-ch10/init-check.json")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"已有结果不覆盖：{args.out}")
    if args.counts_only:
        with torch.device("meta"):
            model = GPT2()
        result = {"params_total": model.n_params(), "params_non_emb": model.n_params(True),
                  "kind": "meta参数计数，无前向运行"}
        assert result["params_total"] == 124_439_808 and result["params_non_emb"] == 85_056_000
    else:
        result = initialization_report(args.batch, args.seq)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(text)
    print(text)

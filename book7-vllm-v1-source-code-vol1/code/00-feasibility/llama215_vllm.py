# llama215_vllm.py —— Book7 ch10 大项目教学档：手写五件套（不取继承捷径）
# 用途：大项目 A 路的教学正身——手写 vllm 模型类的最小完整实现（五件套：__init__/embed_input_ids/
#       forward/compute_logits/load_weights），模块树与权重键位保持 HF 同名（Book3 插槽契约），
#       经 model_class_overrides 注册后与内置 LlamaForCausalLM 同权重对拍（判据 A4：贪心逐字一致）。
#       注：冒烟档（继承捷径版，regA1 底档产出时在位）未另存档——工作区 git 未跟踪期被本档覆写，教训入档；本档为 ch10 教学正身。
# 所属章节：Book7 ch10（§10.6 教学档）；设计书=plan/Book7-vLLM源码剖析上.md §3
# 运行：不直接运行——由 vllm_smoke_reg.py 经 model_class_overrides 以
#       "llama215_vllm:Llama215ForCausalLM" 懒导入（worker 子进程经 PYTHONPATH 可达）。
# 对拍判据（预注册）：A4=与 vllm_smoke.py（C 路）同 prompt 同参数生成逐字一致。
from typing import Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper)
from vllm.model_executor.layers.attention import Attention
from vllm.sequence import IntermediateTensors


class Llama215MLP(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hidden = vllm_config.model_config.hf_config.hidden_size
        inter = vllm_config.model_config.hf_config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden, [inter, inter], bias=False, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(inter, hidden, bias=False,
                                           prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()

    def forward(self, x):                       # (B,n,d) -> (B,n,d)
        gate_up, _ = self.gate_up_proj(x)       # (B,n,2*inter)
        x = self.act_fn(gate_up)                # (B,n,inter)
        x, _ = self.down_proj(x)                # (B,n,d)
        return x


class Llama215Attention(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.qkv_proj = QKVParallelLinear(
            cfg.hidden_size, cfg.hidden_size, cfg.hidden_size,
            total_num_kv_heads=cfg.num_key_value_heads, bias=False,
            prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(cfg.hidden_size, cfg.hidden_size,
                                        bias=False, prefix=f"{prefix}.o_proj")
        self.attn = Attention(
            num_heads=cfg.num_attention_heads,
            head_size=cfg.hidden_size // cfg.num_attention_heads,
            scale=(cfg.hidden_size // cfg.num_attention_heads) ** -0.5,
            num_kv_heads=cfg.num_key_value_heads,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.attn")
        # 与内置 llama.py 同法：rope_parameters 整体透传（transformers 5.x 的 config 子对象）
        self.rotary_emb = get_rope(
            cfg.hidden_size // cfg.num_attention_heads,
            max_position=cfg.max_position_embeddings,
            rope_parameters=getattr(cfg, "rope_parameters", None))

    def forward(self, positions, hidden):       # (B,n,d) -> (B,n,d)
        qkv, _ = self.qkv_proj(hidden)          # (B,n,d+2*d_kv)
        q, k, v = qkv.split([self.attn.num_heads * self.attn.head_size,
                             self.attn.num_kv_heads * self.attn.head_size,
                             self.attn.num_kv_heads * self.attn.head_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v)


class Llama215DecoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.input_layernorm = RMSNorm(cfg.hidden_size,
                                       getattr(cfg, "rms_norm_eps", 1e-5))
        self.self_attn = Llama215Attention(vllm_config=vllm_config,
                                           prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size,
                                                 getattr(cfg, "rms_norm_eps", 1e-5))
        self.mlp = Llama215MLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

    def forward(self, positions, hidden, residual):   # (B,n,d) -> (B,n,d)
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(positions, hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Llama215Model(nn.Module):
    """HF 同名模块树（model.embed_tokens/…/model.norm）——键位契约的载体。"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleDict({
            str(i): Llama215DecoderLayer(vllm_config=vllm_config,
                                         prefix=f"{prefix}.layers.{i}")
            for i in range(cfg.num_hidden_layers)})
        self.norm = RMSNorm(cfg.hidden_size, getattr(cfg, "rms_norm_eps", 1e-5))

    def forward(self, input_ids, positions, intermediate_tensors: Optional[IntermediateTensors] = None):
        hidden = self.embed_tokens(input_ids)               # (B,n,d)
        residual = None
        for i in range(len(self.layers)):
            hidden, residual = self.layers[str(i)](positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden                                        # (B,n,d)


class Llama215ForCausalLM(nn.Module):
    """大项目教学档：手写五件套全实现（门三注册的载体）。

    五件套与本类的对应：__init__(keyword-only vllm_config) / embed_input_ids /
    forward / compute_logits / load_weights——不继承任何模型基类。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.model = Llama215Model(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size,
                                      prefix=f"{prefix}.lm_head")
        self.logits_processor = LogitsProcessor(cfg.vocab_size, **{})

    def embed_input_ids(self, input_ids):                    # (B,n) -> (B,n,d)
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors: Optional[IntermediateTensors] = None):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):                  # (B,n,d) -> (B,n,V)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.model.hf_to_vllm_mapper)

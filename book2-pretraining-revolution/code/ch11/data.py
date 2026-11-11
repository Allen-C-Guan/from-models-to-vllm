# data.py —— Book2第11章：语料字节账、两种块化、同文档编码计数与正规化词基线。
# 工作区激活env.sh后：python code/ch11/data.py --help
# --stages可单独运行某项；WikiText阶段需要datasets，当前修订未冒称重算了历史基线。
# 输出log/book2-ch11；已有报告不覆盖，核算图不覆盖正文图。
import argparse
import hashlib
import math
from pathlib import Path
import json
import os
import time
from collections import Counter

import numpy as np

SEED = 20261002
REPO_ROOT = os.getcwd()
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(REPO_ROOT,"log","cache","tiktoken"))
SCALING_DIR = os.path.join(REPO_ROOT, "log", "book2-scaling")
CH08_DIR = os.path.join(REPO_ROOT, "log", "book2-ch08")
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
FIG_DIR = os.path.join(OUT_DIR, "figures")

DOCS_JSONL = os.path.join(SCALING_DIR, "owt_docs.jsonl")            # LM 主语料（1.15GB / 232,920 篇）
TOK_TXT = os.path.join(SCALING_DIR, "owt_tokenizer_corpus.txt")     # 分词器分片（独立，29MB / 6000 篇）
SP8K_MODEL = os.path.join(CH08_DIR, "sp8k_owt.model")
TOKENS50K = os.path.join(OUT_DIR, "tokens50k.bin")                  # 50257 词表流（train_124m.py --prep-only 产物）
BLOCK = 256                                                          # 训练窗长 s（全书统一）
WT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # 本轮实际查询的固定数据提交


def need(path, hint):
    if not os.path.exists(path):
        raise SystemExit(f"[前置缺失] {path}\n  先跑：{hint}")


# ---------------- ①语料与分片分离的核账 ----------------
def step_corpus():
    """分别数JSONL存储字节、净文本字节/字符；不凭文件名证明内容去重。"""
    need(DOCS_JSONL,"先获取独立LM文档流")
    need(TOK_TXT,"先准备分词器训练文本")
    docs=raw_bytes=chars=0
    h=hashlib.sha256()
    with open(DOCS_JSONL,"rb") as f:
        for raw in f:
            h.update(raw)
            if not raw.strip():continue
            text=json.loads(raw)["text"]
            if not isinstance(text,str):raise ValueError("文档text须为字符串")
            docs+=1;raw_bytes+=len(text.encode("utf-8"));chars+=len(text)
    storage=Path(DOCS_JSONL).stat().st_size
    return {"lm_corpus":{"path":DOCS_JSONL,"docs":docs,"jsonl_bytes":storage,
                         "jsonl_mib":storage/1024**2,"raw_text_utf8_bytes":raw_bytes,
                         "raw_text_characters":chars,"sha256":h.hexdigest(),
                         "license_note":"OpenWebText卡片的CC0是包装层，原网页权利另计"},
            "tokenizer_shard":{"path":TOK_TXT,"bytes":Path(TOK_TXT).stat().st_size,
                              "separation":"历史获取脚本跳过前6000篇用于LM；本函数只核文件，不证明内容无重合"}}


# ---------------- ②两种块化切法对比（教学件） ----------------
def chunking_counts(lengths, block):
    """lengths含每篇EOS；同时给内容库存和右移训练窗口，边界按实际位置去重。"""
    if block<=0 or not lengths or any(n<=0 for n in lengths):
        raise ValueError("block与文档token长度须正")
    total=sum(lengths);content_windows=total//block;pair_windows=(total-1)//block
    boundaries=[];pos=0
    for n in lengths[:-1]:
        pos+=n;boundaries.append(pos)
    content_cross={p//block for p in boundaries if p%block and p<content_windows*block}
    # 输入与右移标签的联合跨度为block+1，整块边界也可能落在最后标签处。
    pair_cross={(p-1)//block for p in boundaries if 0<p<=pair_windows*block}
    per_content=sum(n//block for n in lengths)
    per_pairs=sum((n-1)//block for n in lengths)
    return {"total_tokens":total,"block":block,"docs":len(lengths),
            "A_concat_stream":{"content_windows":content_windows,"content_remainder":total%block,
                "content_utilization":content_windows*block/total,
                "cross_doc_content_windows":len(content_cross),
                "cross_doc_content_fraction":len(content_cross)/content_windows if content_windows else None,
                "training_windows":pair_windows,"training_target_tokens":pair_windows*block,
                "cross_doc_input_or_target_windows":len(pair_cross)},
            "B_per_doc_aligned":{"content_windows":per_content,
                "content_utilization":per_content*block/total,
                "content_discarded_tokens":total-per_content*block,
                "training_windows":per_pairs,"training_target_tokens":per_pairs*block},
            "definition":"内容块只看block个token；训练窗另需右移目标。EOS是词元，未自动重置注意力。"}


def step_chunking(n_docs_demo=2000, block=BLOCK):
    """同一批文档的两种库存/训练窗对照；不假定这就是原GPT2训练管道。"""
    need(SP8K_MODEL,"先准备SP分词器");need(DOCS_JSONL,"先准备LM文档流")
    if n_docs_demo<=0:raise ValueError("演示文档数须正")
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=SP8K_MODEL)
    if sp.eos_id()<0:raise ValueError("SP模型没有有效EOS")
    lengths=[];first=[]
    with open(DOCS_JSONL,encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():continue
            ids=sp.encode(json.loads(raw)["text"],out_type=int,add_eos=True)
            lengths.append(len(ids))
            if len(first)<2:first.append(ids)
            if len(lengths)>=n_docs_demo:break
    rep=chunking_counts(lengths,block)
    junction=first[0][-8:]+first[1][:6] if len(first)>1 else []
    rep.update(requested_docs=n_docs_demo,actual_docs=len(lengths),eos_id=sp.eos_id(),
               junction_ids=junction,junction_decoded=sp.decode(junction))
    return rep


# ---------------- ③token↔word 换算盒（同一文档集、两套词表） ----------------
def step_conversion(n_docs=56000):
    """在完全同一批完整文档上直接数两套编码；两者都加一个EOS/EOT，范围不借用别处元数据。"""
    if n_docs<=0:raise ValueError("文档数须正")
    need(DOCS_JSONL,"先准备LM文档流");need(SP8K_MODEL,"先准备SP分词器")
    import sentencepiece as spm
    import tiktoken
    sp=spm.SentencePieceProcessor(model_file=SP8K_MODEL)
    if sp.eos_id()<0:raise ValueError("SP缺EOS")
    enc=tiktoken.get_encoding("gpt2")
    words=bytes_=gpt=sp_count=docs=0;h=hashlib.sha256()
    with open(DOCS_JSONL,encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            text=json.loads(line)["text"];raw=text.encode("utf-8")
            h.update(len(raw).to_bytes(8,"little"));h.update(raw)
            words+=len(text.split());bytes_+=len(raw)
            gpt+=len(enc.encode(text,disallowed_special=()))+1
            sp_count+=len(sp.encode(text,out_type=int,add_eos=True))
            docs+=1
            if docs>=n_docs:break
    if min(words,bytes_,gpt,sp_count)<=0:raise ValueError("无有效文本可换算")
    return {"requested_docs":n_docs,"docs":docs,"bytes":bytes_,"words":words,
            "document_sequence_sha256":h.hexdigest(),
            "gpt2_50257":{"tokens":gpt,"tokens_per_word":gpt/words,"bytes_per_token":bytes_/gpt},
            "sp8k_8192":{"tokens":sp_count,"tokens_per_word":sp_count/words,"bytes_per_token":bytes_/sp_count},
            "boundary_protocol":"每篇两编码各加一个结束词元；字节只计净文本UTF8，不给人工结束词元添加字节",
            "scope":"完整文档编码统计，不是任意模型eval窗口的精确换算系数；SP内部归一化仍需核对",
            "historical_ch08_3_82":"JSONL包装字节/token，不能当净文本byte换算系数"}


# ---------------- ④WikiText-2 word-level 对照基线 ----------------
def canonical_wikitext(split):
    """按行添加换行分隔；词数与两种token计数均基于同一份文本。"""
    return "\n".join(split["text"])


def word_baselines(train_text, valid_text, gamma=5.0):
    """有限词表+UNK的正规化计数模型；首词用unigram，其余用bigram，均评分完整valid词流。"""
    if not math.isfinite(gamma) or gamma<=0:raise ValueError("gamma须有限正数")
    tr=train_text.split();va_raw=valid_text.split()
    if not tr or not va_raw:raise ValueError("训练与验证词流不能为空")
    counts=Counter(tr);unk="<unk>";added=0
    if counts[unk]==0:counts[unk]=1;added=1
    total=sum(counts.values())
    p1={word:count/total for word,count in counts.items()}
    assert abs(sum(p1.values())-1)<1e-8
    va=[word if word in counts else unk for word in va_raw]
    pairs=Counter(zip(tr,tr[1:]));predecessors=Counter(tr[:-1])
    def prob2(prev,word):
        return (pairs[(prev,word)]+gamma*p1[word])/(predecessors[prev]+gamma)
    nll1=-sum(math.log(p1[word]) for word in va)
    nll2=-math.log(p1[va[0]])-sum(math.log(prob2(prev,word)) for prev,word in zip(va,va[1:]))
    return {"train_words":len(tr),"valid_words":len(va),"valid_oov_rate":sum(w not in counts for w in va_raw)/len(va_raw),
            "vocab_with_unk":len(counts),"unk_pseudocount_added":added,"gamma":gamma,
            "unigram_probability_sum":sum(p1.values()),"nll_unigram":nll1,"nll_bigram":nll2,
            "word_ppl_unigram":math.exp(nll1/len(va)),"word_ppl_bigram_interp":math.exp(nll2/len(va)),
            "scoring_protocol":"同一valid完整词流，首词P1；其余P2；空白切词，OOV映射UNK，不按行重置",
            "not_standard_word_benchmark":"自定义raw-v1计数协议，不冒充预处理word-level榜单"}


def step_wikitext2(revision=WT2_REVISION):
    """CPU计数核验，不训练神经网络；固定数据提交，文本和计数范围一并保存。"""
    import datasets
    import tiktoken
    ds=datasets.load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",revision=revision)
    tr=canonical_wikitext(ds["train"]);va=canonical_wikitext(ds["validation"])
    rep=word_baselines(tr,va)
    enc=tiktoken.get_encoding("gpt2")
    ngpt=len(enc.encode(va,disallowed_special=()))
    rep.update(dataset="Salesforce/wikitext",config="wikitext-2-raw-v1",revision=revision,
               datasets_version=datasets.__version__,text_join="数据行之间补换行符",
               train_text_sha256=hashlib.sha256(tr.encode()).hexdigest(),
               valid_text_sha256=hashlib.sha256(va.encode()).hexdigest(),
               valid_utf8_bytes=len(va.encode()),gpt2_tokens_full_valid=ngpt,
               gpt2_tokens_per_word_full_valid=ngpt/rep["valid_words"],
               ratio_scope="全valid文本的描述统计；不可直接用于只评前若干token的模型损失")
    if os.path.exists(SP8K_MODEL):
        import sentencepiece as spm
        sp=spm.SentencePieceProcessor(model_file=SP8K_MODEL)
        rep["sp8k_tokens_per_word_full_valid"]=len(sp.encode(va))/rep["valid_words"]
    return rep


# ---------------- 图 11.1：管道流 ----------------
def make_figure(rep):
    """核算流程示意，仅用传入结果中的真实范围；输出log，不覆盖正文。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),dpi=300);ax.axis("off")
    rows=[
        ["source",f"LM docs: {rep['corpus']['lm_corpus']['docs']}; tokenizer shard counted separately"],
        ["encoding","SP vocabulary for family/iso; GPT-2 vocabulary for 124M"],
        ["boundaries","append EOS/EOT; marker does not reset causal attention"],
        ["windows","content blocks and shifted input/target windows are different counts"],
        ["evaluation","record text, target positions, tokenizer and context protocol"],
    ]
    table=ax.table(cellText=rows,colLabels=["stage","contract"],colWidths=[.18,.82],bbox=[0,.13,1,.72],cellLoc="left")
    table.auto_set_font_size(False);table.set_fontsize(8)
    for cell in table.get_celld().values():cell.set_edgecolor("#e1e0d9")
    ax.set_title("Document stream, two encodings, explicit counting units")
    fig.tight_layout();os.makedirs(FIG_DIR,exist_ok=True)
    target=Path(FIG_DIR)/"check-11-1-data-pipeline.png"
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,facecolor="white");plt.close(fig)
    return str(target)


def main():
    global OUT_DIR,FIG_DIR,DOCS_JSONL,TOK_TXT,SP8K_MODEL
    ap=argparse.ArgumentParser()
    ap.add_argument("--stages",default="corpus,chunking,conversion,wikitext2")
    ap.add_argument("--docs",default=DOCS_JSONL)
    ap.add_argument("--tokenizer-text",default=TOK_TXT)
    ap.add_argument("--sp-model",default=SP8K_MODEL)
    ap.add_argument("--out-dir",default=OUT_DIR)
    ap.add_argument("--out-name",default="data_pipeline.json")
    ap.add_argument("--revision",default=WT2_REVISION)
    ap.add_argument("--conversion-docs",type=int,default=56000)
    ap.add_argument("--chunk-docs",type=int,default=2000)
    ap.add_argument("--block",type=int,default=BLOCK)
    a=ap.parse_args()
    stages=a.stages.split(",")
    if not stages or any(x not in {"corpus","chunking","conversion","wikitext2"} for x in stages):
        raise ValueError("未知核验阶段")
    if Path(a.out_name).name!=a.out_name or a.out_name in ("",".",".."):
        raise ValueError("out-name只接受文件名")
    OUT_DIR=a.out_dir;FIG_DIR=os.path.join(OUT_DIR,"figures")
    DOCS_JSONL=a.docs;TOK_TXT=a.tokenizer_text;SP8K_MODEL=a.sp_model
    target=Path(OUT_DIR)/a.out_name
    if target.exists():raise FileExistsError(f"已有报告不覆盖：{target}")
    t0=time.perf_counter()
    rep={"meta":{"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                 "stages":stages,"note":"本轮按实际输入核计数，不复写历史实验读数"}}
    for name in stages:
        if name=="corpus":rep[name]=step_corpus()
        elif name=="chunking":rep[name]=step_chunking(a.chunk_docs,a.block)
        elif name=="conversion":rep[name]=step_conversion(a.conversion_docs)
        else:rep[name]=step_wikitext2(a.revision)
    if "corpus" in rep:rep["figure"]=make_figure(rep)
    rep["meta"]["wallclock_sec"]=time.perf_counter()-t0
    payload=json.dumps(rep,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as f:f.write(payload)
    print(payload)
    print(f"[产物] {target}")


if __name__=="__main__":
    main()

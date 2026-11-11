# sample.py —— Book2第11章：同解码配对采样、词法统计和同题ICL。
# 默认加载已有自训断点；仅显式--train-steps才训练，可直接复用train_124m保存的断点。
# 官方权重、自训、随机初始化均比较四种解码，每组重置采样生成器；历史输出不改绑新提示。
# 工作区激活env.sh后：python code/ch11/sample.py --help
# --with-wikitext为显式选做，依赖datasets；只对同一目标范围计S/byte/word。
# 生成全部CPU前向重算，logits fp32、采样概率fp64；输出log且拒绝覆盖。
import argparse
import hashlib
from pathlib import Path
import textwrap
import json
import math
import os
import sys
import time

import numpy as np
import torch

SEED = 20261002
REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch11")
FIG_DIR = os.path.join(OUT_DIR,"figures")
SELF_PT = os.path.join(OUT_DIR, "self124m.pt")
DOCS_JSONL = os.path.join(REPO_ROOT, "log", "book2-scaling", "owt_docs.jsonl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch10"))
import train_124m as t12m          # noqa: E402  复用 fast 档的全部口径常量与数据件（单轨依赖链）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from official_gpt2 import ensure_weights, load_official_into_ours   # noqa: E402  对拍四步的①②③在此复用
from model import GPT2             # noqa: E402  ch10 定版 124M（随机初始化对照侧的机器）

os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(REPO_ROOT, "log", "cache", "tiktoken"))

GEN_PROMPT = "The meaning of life is"     # 采样演示 prompt（与对拍 7-prompt 分开：这里要看的是开放续写）
GEN_TOKENS = 40                           # 每条生成的续写长度


def get_tok():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


# ---------------- 生成器（全量重算口径） ----------------
@torch.no_grad()
def generate(model,enc,prompt,n_new=GEN_TOKENS,temperature=1.0,top_k=0,generator=None,device="cpu"):
    """T=0贪心；T>0从明确top-k集合采样，平分时采用topk返回的k个索引。

    每步重算(1,n)->(1,n,V)。输入/输出长度须在上下文内；退出恢复原训练模式。
    """
    if not isinstance(n_new,int) or n_new<0:raise ValueError("n_new须为非负整数")
    if not math.isfinite(temperature) or temperature<0:raise ValueError("温度须为有限非负数")
    if not isinstance(top_k,int) or top_k<0:raise ValueError("top_k须为非负整数")
    encoded=enc.encode(prompt,disallowed_special=())
    if not encoded:raise ValueError("生成需要非空前缀")
    if len(encoded)+max(0,n_new-1)>model.block_size:raise ValueError("超过上下文上限")
    ids=torch.tensor([encoded],dtype=torch.long,device=device)
    before=model.training;model.eval();out=[]
    try:
        for _ in range(n_new):
            logits,_=model(ids);last=logits[0,-1].double()
            if not torch.isfinite(last).all():raise FloatingPointError("生成logits非有限")
            if top_k>len(last):raise ValueError("top_k超过词表大小")
            if temperature==0:
                nxt=int(last.argmax())
            else:
                values,index=last.topk(top_k) if top_k else (last,torch.arange(len(last),device=last.device))
                # 减最大值不改变softmax；float64概率计算减少极小正温度的溢出风险。
                probs=torch.softmax((values-values.max())/temperature,dim=-1)
                choice=torch.multinomial(probs,1,generator=generator)
                nxt=int(index[choice].item())
            out.append(nxt)
            ids=torch.cat([ids,torch.tensor([[nxt]],dtype=torch.long,device=device)],dim=1)
        return out
    finally:model.train(before)


# ---------------- 自训侧：复现 fast 档并保留权重 ----------------
def train_self(steps,device,out_name="self_sampling"):
    """显式训练才调用；复用同一训练执行函数，保留的断点可直接采样，无需另写不一致循环。"""
    result=t12m.train_124m(device,out_name=out_name,steps=steps,bootstrap_draws=0)
    if not result["completed"]:raise RuntimeError("自训未完成，不用失败权重做正式对照")
    model,info=load_self("cpu",result["checkpoint"])
    return model,{"run":result,"checkpoint":info,
                  "schedule_note":"总日程等于所选steps，缩短steps不等于只截取原6104步轨迹前缀"}


def load_self(device,checkpoint_path=None):
    """读取同一ch04工厂的权重；新版断点保留训练身份，旧裸state_dict只标来源未知。"""
    from warmup_ablation import GPT
    path=Path(checkpoint_path or SELF_PT)
    if not path.is_file():raise FileNotFoundError(path)
    payload=torch.load(path,map_location=device,weights_only=True)
    if isinstance(payload,dict) and "model" in payload:
        config=payload.get("config")
        if not config:raise ValueError("新版断点缺模型配置")
        state=payload["model"]
        provenance=payload.get("run_identity")
        steps=payload.get("step")
    else:
        config=dict(n_layer=t12m.N_LAYER,n_embd=t12m.N_EMBD,n_head=t12m.N_HEAD,
                    vocab=t12m.VOCAB,block_size=t12m.WPE)
        state=payload;provenance=None;steps=None
    model=GPT(**config).to(device).eval()
    model.load_state_dict(state,strict=True)
    info={"file":str(path),"sha256":t12m.file_identity(path)["sha256"],"config":config,"steps":steps,
          "run_identity":provenance,"legacy_note":"裸权重缺训练身份，不推断其数据与日程" if provenance is None else None}
    return model,info


# ---------------- 词法统计 ----------------
def lexical_stats(text,ref_bigrams):
    """空白分词、大小写和标点照留；无词或无词对时比例未定义，不伪报0。"""
    words=text.split();pairs=list(zip(words,words[1:]));n=len(text)
    return {"n_words":len(words),"n_bigrams":len(pairs),
            "mean_word_len":sum(map(len,words))/len(words) if words else None,
            "space_rate":text.count(" ")/n if n else None,
            "alpha_rate":sum(c.isalpha() for c in text)/n if n else None,
            "type_token_ratio":len(set(words))/len(words) if words else None,
            "bigram_hit_rate":sum(p in ref_bigrams for p in pairs)/len(pairs) if pairs else None,
            "distinct_bigram_ratio":len(set(pairs))/len(pairs) if pairs else None}


def build_ref_bigrams(n_docs=2000,max_words=300_000,return_info=False):
    """逐文档提取词对；不在文档连接处造一对词，严格截到max_words。"""
    if min(n_docs,max_words)<=0:raise ValueError("文档/词预算须正")
    pairs=set();used=docs=0
    with open(DOCS_JSONL,encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():continue
            if docs>=n_docs or used>=max_words:break
            words=json.loads(raw)["text"].split()[:max_words-used]
            pairs.update(zip(words,words[1:]));used+=len(words);docs+=1
    info={"actual_docs":docs,"actual_words":used,"distinct_bigrams":len(pairs),
          "protocol":"文档内空白词对，大小写/标点照留；无跨文档拼接词对"}
    return (pairs,info) if return_info else pairs


# ---------------- WikiText-2 换算盒演示（自训模型跨语料诚实口径） ----------------
@torch.no_grad()
def wt2_eval(model,enc,device,n_windows=24,block=256,text=None):
    """按实际被计分的目标token累加S；字节/词分母都取同一范围，不套全valid比率。"""
    if n_windows<=0 or block<=0:raise ValueError("窗口数和长度须正")
    source={"kind":"provided_text"}
    if text is None:
        from datasets import load_dataset
        revision="b08601e04326c79dfdd32d625aee71d232d685c3"
        ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",revision=revision)
        text="\n".join(ds["validation"]["text"])
        source={"dataset":"Salesforce/wikitext","config":"wikitext-2-raw-v1","revision":revision}
    ids=enc.encode(text,disallowed_special=())
    windows=min(n_windows,(len(ids)-1)//block)
    if windows==0:raise ValueError("文本不够一个完整输入/目标窗")
    before=model.training;model.eval();nll=0.
    try:
        for i in range(windows):
            x=torch.tensor([ids[i*block:(i+1)*block]],dtype=torch.long,device=device)
            y=torch.tensor([ids[i*block+1:(i+1)*block+1]],dtype=torch.long,device=device)
            _,loss=model(x,y)
            value=float(loss)
            if not math.isfinite(value):raise FloatingPointError("WT2 loss非有限")
            nll+=value*block
    finally:model.train(before)
    count=windows*block;pieces=[enc.decode_single_token_bytes(i) for i in ids]
    whole=b"".join(pieces);raw=text.encode("utf-8")
    if whole!=raw:raise ValueError("本函数字节分母要求无损还原原文本的编码")
    start=len(pieces[0]);end=sum(map(len,pieces[:count+1]))
    segment=raw[start:end];word_count=None;reason=None
    try:
        prefix=raw[:start].decode("utf-8");scored=segment.decode("utf-8");suffix=raw[end:].decode("utf-8")
        left_ok=not prefix or prefix[-1].isspace() or (scored and scored[0].isspace())
        right_ok=not suffix or suffix[0].isspace() or (scored and scored[-1].isspace())
        if left_ok and right_ok and scored.split():word_count=len(scored.split())
        else:reason="评分范围切在词内，或没有完整词；不报告精确word PPL"
    except UnicodeDecodeError:reason="评分范围切在UTF8字符内；保留byte分母，不猜词数"
    def exp_or_none(value):return math.exp(value) if value<700 else None
    return {"source":source,"text_sha256":hashlib.sha256(raw).hexdigest(),
            "requested_windows":n_windows,"actual_windows":windows,"block":block,
            "target_token_range":[1,count+1],"target_byte_range":[start,end],
            "n_tokens_eval":count,"n_bytes_eval":len(segment),"n_words_eval":word_count,
            "total_nll":nll,"L_token_nats":nll/count,"PPL_token":exp_or_none(nll/count),
            "nats_per_byte":nll/len(segment),"bits_per_byte":nll/len(segment)/math.log(2),
            "PPL_word_converted":exp_or_none(nll/word_count) if word_count else None,
            "word_boundary_note":reason,
            "protocol":"首token只作上下文；各block重置上下文，S及分母来自相同目标范围；非标准WT2榜单"}


# ---------------- ICL 探针（选做；官方权重本地前向，不调 API） ----------------
ICL_TASKS = [
    {"task": "capital few-shot",
     "zero": "The capital of Italy is",
     "few": "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"},
    {"task": "antonym few-shot",
     "zero": "The opposite of big is",
     "few": "The opposite of hot is cold. The opposite of up is down. The opposite of big is"},
    {"task": "EN->FR pair (GPT-2 paper style)",
     "zero": "English: dog. French:",
     "few": "English: hello. French: bonjour.\nEnglish: cheese. French: fromage.\nEnglish: dog. French:"},
]


@torch.no_grad()
def icl_probe(model, enc, device):
    """每个任务取两个口径：zero-shot vs few-shot，各记（末位 top-1 token 及其概率）+（贪心续写 8 token）。
    当前提示固定相同末问题；历史提示中的hot/big、cheese/dog不配对，旧结果不得挪给本版。"""
    rows = []
    for t in ICL_TASKS:
        row = {"task": t["task"]}
        for tag, p in (("zero_shot", t["zero"]), ("few_shot", t["few"])):
            ids = torch.tensor([enc.encode(p)], dtype=torch.long, device=device)
            logits, _ = model(ids)
            last = logits[0, -1, :]
            nxt = int(last.argmax())
            p_val = float(torch.softmax(last, -1)[nxt])
            cont = generate(model, enc, p, n_new=8, temperature=0.0, device=device)
            row[tag] = {"prompt": p, "greedy_next": enc.decode([nxt]), "p": round(p_val, 4),
                        "greedy_8tok": enc.decode(cont)}
        rows.append(row)
        print(f"  [ICL] {t['task']}: zero -> {row['zero_shot']['greedy_8tok']!r} | "
              f"few -> {row['few_shot']['greedy_8tok']!r}", flush=True)
    return rows


# ---------------- 图 11.3 ----------------
def make_figure_real(rep):
    """当前采样结果按明确decode_key分组；旧历史图不由本函数覆盖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(12,5.8),dpi=300,gridspec_kw={"width_ratios":[1.5,1.]})
    left.axis("off")
    rows=[[s["model"],s["decode_key"],textwrap.fill(s["text"].replace("\n"," ")[:72],48)] for s in rep["samples"]]
    table=left.table(cellText=rows,colLabels=["model","decode","excerpt (full text in JSON)"],
                     colWidths=[.14,.16,.7],cellLoc="left",bbox=[0,.02,1,.95])
    table.auto_set_font_size(False);table.set_fontsize(6.8)
    for cell in table.get_celld().values():cell.set_edgecolor("#e1e0d9")
    keys=("greedy","temp1","temp07","topk40")
    for j,(name,color) in enumerate((("official","#1baf7a"),("self","#eb6834"),("random-init","#898781"))):
        values=[]
        for key in keys:
            item=next(x for x in rep["lexical"] if x.get("model")==name and x.get("decode_key")==key)
            values.append(item["bigram_hit_rate"])
        xs=np.arange(4)+(j-1)*.24
        for x,value in zip(xs,values):
            if value is None:right.text(x,0,"NA",ha="center",fontsize=7)
            else:right.bar(x,value,width=.22,color=color)
        right.plot([],[],color=color,lw=6,label=name)
    right.set_xticks(range(4),keys);right.set_ylim(0,1.08)
    right.set_ylabel("word-bigram hit rate (observed samples)")
    right.set_title("Hold decoding fixed when comparing models")
    right.legend(frameon=False,fontsize=8);right.grid(axis="y",color="#e1e0d9",lw=.5);right.set_axisbelow(True)
    fig.tight_layout();Path(FIG_DIR).mkdir(parents=True,exist_ok=True)
    target=Path(FIG_DIR)/"check-11-3-sampling.png"
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,facecolor="white");plt.close(fig)


def main():
    global OUT_DIR,FIG_DIR,DOCS_JSONL
    ap=argparse.ArgumentParser(description="显式加载自训权重后采样；只在--train-steps给定时训练")
    ap.add_argument("--train-steps",type=int,default=None)
    ap.add_argument("--skip-train",action="store_true",help="兼容入口：缺权重时报错，不会自动转为长训练")
    ap.add_argument("--train-only",action="store_true")
    ap.add_argument("--self-checkpoint",type=Path)
    ap.add_argument("--data-dir",default=t12m.OUT_DIR)
    ap.add_argument("--out-dir",default=OUT_DIR)
    ap.add_argument("--out-name",default="sampling.json")
    ap.add_argument("--docs",default=DOCS_JSONL)
    ap.add_argument("--snapshot-dir")
    ap.add_argument("--revision")
    ap.add_argument("--with-wikitext",action="store_true")
    ap.add_argument("--device",choices=["cpu","mps"],default="mps" if torch.backends.mps.is_available() else "cpu")
    a=ap.parse_args()
    if a.train_steps is not None and (a.skip_train or a.self_checkpoint):
        raise ValueError("显式训练与加载已有权重不能同时选择")
    if a.train_only and a.train_steps is None:raise ValueError("train-only需要给出train-steps")
    if Path(a.out_name).name!=a.out_name or a.out_name in ("",".",".."):raise ValueError("out-name只接受文件名")
    OUT_DIR=a.out_dir;DOCS_JSONL=a.docs;FIG_DIR=os.path.join(OUT_DIR,"figures",Path(a.out_name).stem)
    target=Path(OUT_DIR)/a.out_name
    if target.exists():raise FileExistsError(target)
    t12m.OUT_DIR=a.data_dir
    if a.train_steps is not None:
        self_model,train_meta=train_self(a.train_steps,a.device,Path(a.out_name).stem+"_self")
    else:
        checkpoint=a.self_checkpoint or Path(a.data_dir)/"ckpt_train124_fast.pt"
        self_model,info=load_self("cpu",checkpoint)
        train_meta={"checkpoint":info,"note":"仅加载，不训练"}
    if a.train_only:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x") as f:json.dump(train_meta,f,ensure_ascii=False,indent=2,allow_nan=False)
        return
    enc=get_tok()
    if self_model.wte.num_embeddings!=enc.n_vocab:raise ValueError("自训词表与gpt2编码器不符")
    dl=ensure_weights(a.revision,a.snapshot_dir);official,audit=load_official_into_ours(dl["snapshot_dir"])
    torch.manual_seed(SEED);random_model=GPT2().eval()
    refs,ref_info=build_ref_bigrams(return_info=True)
    rep={"meta":{"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"seed":SEED,
                 "generation_device":"cpu","logits_dtype":"fp32","sampling_probability_dtype":"float64",
                 "self_train":train_meta,"official_snapshot":dl,"prompt":GEN_PROMPT,"gen_tokens":GEN_TOKENS,
                 "prompt_protocol":"same_question_v2；不改绑历史输出","reference_bigrams":ref_info},
         "samples":[],"lexical":[],"official_audit":audit}
    settings=[("greedy",0.,0),("temp1",1.,0),("temp07",.7,0),("topk40",1.,40)]
    for name,model in (("official",official),("self",self_model),("random-init",random_model)):
        model=model.to("cpu").eval()
        for key,temp,k in settings:
            gen=torch.Generator(device="cpu").manual_seed(SEED)
            ids=generate(model,enc,GEN_PROMPT,temperature=temp,top_k=k,generator=gen,device="cpu")
            text=enc.decode(ids)
            rep["samples"].append({"model":name,"decode_key":key,"temperature":temp,"top_k":k,
                                   "seed":SEED,"text":text,"token_ids":ids})
            rep["lexical"].append({"model":name,"decode_key":key,**lexical_stats(text,refs)})
    with open(DOCS_JSONL,encoding="utf-8") as f:reference=json.loads(f.readline())["text"][:1200]
    rep["corpus_reference"]={**lexical_stats(reference,refs),
         "note":"参考集合内文本，不是独立测试，也不进入模型对照柱"}
    if a.with_wikitext:rep["wt2_box"]=wt2_eval(self_model.to("cpu"),enc,"cpu")
    rep["icl"]=icl_probe(official.to("cpu"),enc,"cpu")
    payload=json.dumps(rep,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    target.parent.mkdir(parents=True,exist_ok=True)
    make_figure_real(rep)
    with target.open("x") as f:f.write(payload)
    print(f"[当前采样记录] {target}")


if __name__ == "__main__":
    main()

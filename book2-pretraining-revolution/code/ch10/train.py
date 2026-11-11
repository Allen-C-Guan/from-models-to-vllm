# train.py —— Book2 ch10 大项目 I 定版训练循环（CLI）
# 所属章节：Book2 第 10 章 §10.3/10.5（drafts/Book2-预训练革命/ch10-大项目I.md 图 10.2/10.3 数据源）
# 演进：Book1 -> ch4/ch8前身 -> 本章循环；ch11训练自备循环，官方权重对拍才复用本章模型。
# 配方：AdamW(0.9, 0.95, wd 0.1) + grad clip 1.0 + cosine 退火（warmup 线性升、降到 0.1x 峰值）+ 可选 bf16 autocast；
#   eval 协议 = held-out 段「一次通过、窗口不重叠」（第 4 章 47 遍记忆化教训的修正）。
# 任务：--task char（6L/384d/6H 字符级热身，参考nanoGPT主要尺寸，本书非嵌入10.65M；配方与计数边界见正文）
#       --task smoke124（124M 冒烟：tiny shakespeare 自训 sp-8k 编码——分词器在独立前 10% 分片上训练、与 LM 语料
#                        分离；id<8192 天然落在 50257 词表内，参数 assert 不变；只验循环健康不追指标）
#       --parity-hf（验收②：随机权重同 config 对拍 HF，CPU fp32）
# 运行方式：python train.py --task char --steps 1000 | --task smoke124 --steps 150 --bf16 | --parity-hf
# 产物：log/book2-ch10/{out-name}.json（meta/曲线/计时/内存峰值/语料一元账 corpus_stats）
import argparse
import datetime
import hashlib
import math
from contextlib import nullcontext
from pathlib import Path
import json
import os
import threading
import time
import urllib.request

import numpy as np
import torch

from model import GPT2, GPT2Config, GPT2_124M

SEED = 20261002                       # 全书统一种子
REPO_ROOT = os.getcwd()
OUT_DIR = os.path.join(REPO_ROOT, "log", "book2-ch10")
DATA_PATH = os.path.join(OUT_DIR, "data", "tiny_shakespeare.txt")
FALLBACK_DATA = os.path.join(REPO_ROOT, "log", "book2-feasibility", "tiny_shakespeare.txt")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

TASKS = {  # 各任务默认档（均可被 CLI 覆盖）
    "char": dict(n_layer=6, n_embd=384, n_head=6, block=256, batch=64, lr=1e-3,
                 warmup=100, steps=1000, schedule_total=5000, eval_every=250),
    "smoke124": dict(**GPT2_124M, block=256, batch=8, lr=1e-4,
                     warmup=10, steps=150, schedule_total=150, eval_every=50),
}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_corpus() -> str:
    """tiny shakespeare（公版莎剧子集；char-rnn 仓库代码 MIT、数据文件无专门授权条款）。"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        if os.path.exists(FALLBACK_DATA):                   # 复用探针/第 4 章已下载副本（拷贝，不动原文件）
            import shutil
            shutil.copyfile(FALLBACK_DATA, DATA_PATH)
        else:
            print("下载 tiny shakespeare …", flush=True)
            partial = DATA_PATH + ".download"
            if os.path.exists(partial):
                raise FileExistsError(f"未完成的下载已存在，先检查：{partial}")
            try:
                urllib.request.urlretrieve(SHAKESPEARE_URL, partial)
                if os.path.getsize(partial) == 0:
                    raise ValueError("下载得到空语料")
                os.replace(partial, DATA_PATH)
            except BaseException:
                # 保留下载现场，下一次不把残件当完整语料。
                raise
    return DATA_PATH


def load_char_ids():
    """字符级编码：全文 -> 排序去重字符表 -> id 流；位置切分 train 90% / val 10%。"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * 0.9)
    return ids[:n], ids[n:], len(chars), {"kind": "char", "vocab": len(chars)}


def load_sp8k_ids():
    """先用文本前10%训练独立SP；剩余文本整体编码后，按token数90/10切LM段。"""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = os.path.join(OUT_DIR, "sp8k_shakespeare")
    text = Path(DATA_PATH).read_text()
    n_tok = int(len(text) * .1)
    if n_tok < 1:
        raise ValueError("语料不足以切出分词器训练段")
    tok_model, meta_path = Path(prefix+".model"), Path(prefix+".meta.json")
    identity = {"corpus_sha256": file_sha(DATA_PATH),
                "tokenizer_text_sha256": hashlib.sha256(text[:n_tok].encode()).hexdigest(),
                "tokenizer_text_chars": n_tok, "model_type": "bpe", "vocab": 8192,
                "byte_fallback": True, "character_coverage": 1.0,
                "sentencepiece_version": spm.__version__}
    if tok_model.exists():
        if not meta_path.exists():
            raise ValueError("旧分词器缺身份记录；请另选out-dir，不覆盖旧模型")
        saved = json.loads(meta_path.read_text())
        if saved.get("identity") != identity or saved.get("model_sha256") != file_sha(tok_model):
            raise ValueError("分词器与当前语料/规格不符，请另选out-dir")
    else:
        if any(Path(prefix+ext).exists() for ext in (".vocab",".txt",".meta.json")):
            raise FileExistsError("分词器旁件已存在，先检查或另选out-dir")
        Path(prefix+".txt").write_text(text[:n_tok])
        spm.SentencePieceTrainer.train(input=prefix+".txt", model_prefix=prefix,
                                      model_type="bpe", vocab_size=8192,
                                      byte_fallback=True, character_coverage=1.0)
        meta_path.write_text(json.dumps({"identity": identity, "model_sha256": file_sha(tok_model)},
                                       ensure_ascii=False, indent=2)+"\n")
    sp = spm.SentencePieceProcessor(model_file=str(tok_model))
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(tok_model.read_bytes())
    if (sp.GetPieceSize() != 8192 or not proto.trainer_spec.byte_fallback
            or proto.trainer_spec.model_type != sentencepiece_model_pb2.TrainerSpec.BPE):
        raise ValueError("分词器实际规格不是8192 BPE + byte_fallback")
    ids = torch.tensor(sp.EncodeAsIds(text[n_tok:]), dtype=torch.long)
    cut = int(len(ids)*.9)
    info = {"kind": "sp-8k(bpe, byte_fallback)", "identity": identity,
            "model_sha256": file_sha(tok_model), "vocab": sp.GetPieceSize(),
            "split": "前10%字符训tokenizer；剩余整体编码后90/10 token切分",
            "seed_note": "PyTorch种子不等于SentencePiece训练种子；复用以实际模型哈希为准"}
    return ids[:cut], ids[cut:], sp.GetPieceSize(), info


def get_batch(split_ids, block, batch, device):
    """随机输入ids[a:a+block]和右移标签ids[a+1:a+block+1]，均(B,n)。"""
    if block <= 0 or batch <= 0 or len(split_ids) < block+1:
        raise ValueError("批量/窗口须正且至少需要block+1个token")
    ix = torch.randint(len(split_ids)-block, (batch,))
    x = torch.stack([split_ids[i:i+block] for i in ix]).to(device)
    y = torch.stack([split_ids[i+1:i+block+1] for i in ix]).to(device)
    return x, y


def corpus_stats(train_ids, val_ids, vocab):
    """训练段经验一元熵及覆盖率；不是验证损失下界，也不替代验证集交叉熵。"""
    cnt = torch.bincount(train_ids, minlength=vocab).double()
    p = cnt / cnt.sum()
    p = p[p > 0]
    return {"train_tokens": int(train_ids.numel()), "val_tokens": int(val_ids.numel()),
            "train_active_types": int((cnt > 0).sum()),
            "train_unigram_entropy_nat": round(float(-(p * p.log()).sum()), 2),
            "uniform_entropy_lnV": round(float(torch.log(torch.tensor(float(vocab)))), 2)}


@torch.no_grad()
def eval_once_through(model, val_ids, block, batch, device, bf16=False):
    """各目标位置至多计一次，完整窗口的token加权均值；返回标量并恢复原模式。"""
    if block <= 0 or batch <= 0 or len(val_ids) < block+1:
        raise ValueError("验证段至少需要一个完整输入/目标窗口")
    was_training = model.training
    model.eval()
    total_loss, total_tok = 0.0, 0
    starts = list(range(0, len(val_ids)-block, block))
    try:
        for j in range(0, len(starts), batch):
            chunk = starts[j:j+batch]
            xb = torch.stack([val_ids[i:i+block] for i in chunk]).to(device)
            yb = torch.stack([val_ids[i+1:i+block+1] for i in chunk]).to(device)
            ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if bf16 else nullcontext()
            with ctx:
                _, loss = model(xb, yb)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("验证loss非有限")
            total_loss += loss.item()*xb.numel()
            total_tok += xb.numel()
    finally:
        model.train(was_training)
    return total_loss/total_tok


class MpsMemSampler(threading.Thread):
    """current张量占用的轮询峰值；不含未用缓存，短峰仍可能漏采。"""
    def __init__(self, interval=.03):
        super().__init__(daemon=True)
        self.interval, self.peak, self.error = interval, 0.0, None
        self._stop_evt = threading.Event()
    def sample(self):
        self.peak = max(self.peak, torch.mps.current_allocated_memory()/1024**2)
    def run(self):
        try:
            while not self._stop_evt.wait(self.interval):
                self.sample()
        except BaseException as exc:
            self.error = exc
            self._stop_evt.set()
    def stop(self):
        self._stop_evt.set()
        self.join()
        if self.error is not None:
            raise RuntimeError("MPS采样失败，不报告有效峰值") from self.error
        self.sample()
        return round(self.peak,1)


def lr_at(step, peak, warmup, total, min_ratio=0.1):
    """lr 日程：warmup 线性升到 peak，之后余弦降到 min_ratio*peak（ch4/ch8 同一函数的定版）。"""
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_ratio * peak + (1 - min_ratio) * peak * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))


def run_task(args):
    """从头运行一个任务；成功/失败均落独立报告，不支持断点续训。"""
    global OUT_DIR, DATA_PATH
    OUT_DIR = str(getattr(args, "out_dir", None) or OUT_DIR)
    if getattr(args, "data_path", None) is not None:
        DATA_PATH = str(args.data_path)
        if not Path(DATA_PATH).is_file():
            raise FileNotFoundError(DATA_PATH)
    else:
        DATA_PATH = os.path.join(OUT_DIR, "data", "tiny_shakespeare.txt")
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if device not in ("cpu","mps") or (device=="mps" and not torch.backends.mps.is_available()):
        raise ValueError("本脚本仅CPU/MPS；所选MPS须实际可用")
    cfgd = dict(TASKS[args.task])
    steps = cfgd["steps"] if args.steps is None else args.steps
    eval_every = cfgd["eval_every"] if args.eval_every is None else args.eval_every
    total = cfgd["schedule_total"] if args.schedule_total is None else args.schedule_total
    if min(steps,eval_every,total) <= 0 or total <= cfgd["warmup"] or steps > total:
        raise ValueError("步数/评估间隔须正，完整日程须长于warmup，运行步数不能超日程")
    name = args.out_name or args.task
    if Path(name).name != name or name in ("",".",".."):
        raise ValueError("out-name只接受文件名，不含路径")
    out = Path(OUT_DIR)/(name if name.endswith(".json") else name+".json")
    if out.exists():
        raise FileExistsError(f"已有结果不覆盖：{out}")
    torch.manual_seed(args.seed)
    ensure_corpus()
    train_ids,val_ids,vocab,tok_info = load_char_ids() if args.task=="char" else load_sp8k_ids()
    if min(len(train_ids),len(val_ids)) < cfgd["block"]+1:
        raise ValueError("训练/验证段均须至少容纳一个完整窗口")
    cfg = GPT2Config(vocab_size=50257 if args.task=="smoke124" else vocab,
                    n_positions=1024 if args.task=="smoke124" else cfgd["block"],
                    n_layer=cfgd["n_layer"],n_embd=cfgd["n_embd"],n_head=cfgd["n_head"])
    model = GPT2(cfg).to(device)
    if args.task=="smoke124":
        assert model.n_params()==124_439_808
    opt = torch.optim.AdamW(model.parameters(),lr=cfgd["lr"],betas=(.9,.95),weight_decay=.1)
    x0,y0=get_batch(val_ids,cfgd["block"],min(cfgd["batch"],8),device)
    with torch.no_grad():
        _,l0=model(x0,y0)
    if not bool(torch.isfinite(l0)):
        raise FloatingPointError("初始化检查loss非有限，未进入训练")
    report={"status":"running","meta":{
        "date_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed":args.seed,"task":args.task,"device":device,
        "dtype":"bf16-autocast" if args.bf16 else "fp32","torch":str(torch.__version__),
        "corpus_bytes":os.path.getsize(DATA_PATH),"corpus_sha256":file_sha(DATA_PATH),
        "code_sha256":{name:file_sha(Path(__file__).with_name(name)) for name in ("train.py","model.py")},
        "tokenizer":tok_info,"model_config":vars(cfg),"model_vocab":cfg.vocab_size,
        "n_params_total":model.n_params(),"n_params_non_embedding":model.n_params(True),
        "block":cfgd["block"],"batch":cfgd["batch"],"lr":cfgd["lr"],"steps":steps,
        "warmup":cfgd["warmup"],"schedule_total":total,"eval_every":eval_every,
        "optimizer":"AdamW(.9,.95,wd=.1，全参数)+clip1+cosine->.1x",
        "eval_protocol":"完整目标窗口一次通过，token加权，固定窗口不保证独立",
        "eval_tokens":((len(val_ids)-1)//cfgd["block"])*cfgd["block"],
        "eval_tail_unscored":(len(val_ids)-1)%cfgd["block"],
        "init_loss_check":l0.item(),"init_loss_check_dtype":"fp32",
        "init_loss_expected_lnV":math.log(cfg.vocab_size),
        "restart_note":"从头运行；不保存权重断点"},
        "corpus_stats":corpus_stats(train_ids,val_ids,vocab),"curve":[]}
    mem=MpsMemSampler() if device=="mps" else None
    if mem:
        torch.mps.synchronize();mem.sample();mem.start()
    t0=time.perf_counter();t_train=0.;run_sum=0.;run_n=0;completed=0
    failure=None
    try:
        for step in range(1,steps+1):
            for g in opt.param_groups:
                g["lr"]=lr_at(step,cfgd["lr"],cfgd["warmup"],total)
            ts=time.perf_counter()
            x,y=get_batch(train_ids,cfgd["block"],cfgd["batch"],device)
            opt.zero_grad(set_to_none=True)
            ctx=torch.autocast(device_type=device,dtype=torch.bfloat16) if args.bf16 else nullcontext()
            with ctx:
                _,loss=model(x,y)
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("训练loss非有限，未更新参数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if device=="mps":
                torch.mps.synchronize()
            t_train+=time.perf_counter()-ts
            completed=step;run_sum+=loss.item();run_n+=1
            if step%eval_every==0 or step==steps:
                vl=eval_once_through(model,val_ids,cfgd["block"],8,device,args.bf16)
                report["curve"].append({"step":step,"train_loss":run_sum/run_n,"val_loss":vl})
                print(f"step {step} train {run_sum/run_n:.4f} val {vl:.4f}",flush=True)
                run_sum=0.;run_n=0
        report["status"]="complete"
    except BaseException as exc:
        failure=exc;report["status"]="failed";report["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        if mem:
            try:
                torch.mps.synchronize()
                report["mps_peak_mib"]=mem.stop()
            except BaseException as exc:
                if mem.is_alive():
                    mem._stop_evt.set();mem.join()
                failure=failure or exc;report["status"]="failed";report["memory_error"]=str(exc)
        else:
            report["mps_peak_mib"]=None
    elapsed=time.perf_counter()-t0
    report.update(completed_steps=completed,sec_total=elapsed,
                  sec_per_step=elapsed/completed if completed else None,
                  sec_train_only_per_step=t_train/completed if completed else None,
                  timing_note="同步训练步含数据搬运和有限性检查；总时间还含评估和采样停止，不含模型/数据准备和报告写盘")
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        f.write(payload)
    print(f"[{report['status']}] {out}")
    if failure:
        raise failure
    return report


def parity_hf(small=False, seed=SEED, out_path=None):
    """验收②：随机权重同 config 对拍 HF（CPU fp32）——GPT2LMHeadModel(GPT2Config()) 随机初始化
    （不下权重），state_dict 按键映射表搬进手写模型（Conv1D 转置 / tied 补绑 / 前缀剥掉），
    同输入的logits一致只支持已测配置与输入，有限对照不是所有输入上的数学证明。"""
    from transformers import GPT2Config as HFCfg, GPT2LMHeadModel
    import transformers
    if out_path is not None and Path(out_path).exists():
        raise FileExistsError(f"已有对拍结果不覆盖：{out_path}")
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=31,n_positions=16,n_layer=2,n_head=2,n_embd=16) if small else GPT2Config()
    ours = GPT2(cfg).eval()
    cfg_hf = HFCfg(vocab_size=cfg.vocab_size,n_positions=cfg.n_positions,
                   n_layer=cfg.n_layer,n_head=cfg.n_head,n_embd=cfg.n_embd,
                   resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,activation_function="gelu_new",
                   bos_token_id=cfg.vocab_size-1,eos_token_id=cfg.vocab_size-1)
    cfg_hf._attn_implementation = "eager"                  # v5：注意力实现挂在 config（默认 sdpa；eager 与手写路径同型）
    hf = GPT2LMHeadModel(cfg_hf).eval()
    n_ours, n_hf = ours.n_params(), sum(p.numel() for p in hf.parameters())
    assert n_ours == n_hf
    if not small:
        assert n_ours == 124_439_808
    print(f"参数量：本书 {n_ours:,} vs HF {n_hf:,}")

    sd, n_t = {}, 0
    for k, v in hf.state_dict().items():
        if k.endswith((".attn.bias", ".attn.masked_bias")) or k.startswith("lm_head."):
            continue                                        # 仅跳过已确认的掩码键及共享头别名；磁盘文件是否去重另查。
        k2 = k.replace("transformer.", "").replace("h.", "blocks.")
        if k2.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            v = v.t()                                       # Conv1D (in,out) -> Linear (out,in)
            n_t += 1
        sd[k2] = v
    sd["head.weight"] = sd["wte.weight"]                    # 加载所需共享别名键；参数共享已由构造函数建立
    ours.load_state_dict(sd, strict=True)
    print(f"键映射搬移完成（转置 {n_t} 处 Conv1D 权重；strict=True 无缺键/多键即通过）")

    g = torch.Generator().manual_seed(seed)                # 固定输入（随机 token；真实文本对拍归 ch11）
    ids = torch.randint(0, cfg.vocab_size, (2, 8) if small else (4, 128), generator=g)
    with torch.no_grad():
        lg_ours, _ = ours(ids)
        lg_hf = hf(input_ids=ids).logits
    d = (lg_ours - lg_hf).abs()
    top5_o = lg_ours.topk(5, dim=-1).indices
    top5_h = lg_hf.topk(5, dim=-1).indices
    top5_hit = (top5_o.unsqueeze(-1) == top5_h.unsqueeze(-2)).any(-1).float().mean().item()
    am = (lg_ours.argmax(-1) == lg_hf.argmax(-1)).float().mean().item()
    print(f"logits 对拍：max|Δ| = {d.max().item():.3e}   mean|Δ| = {d.mean().item():.3e}")
    print(f"argmax 一致率 = {am:.4f}   top-5 命中率 = {top5_hit:.4f}")
    passed = bool(torch.isfinite(d).all()) and d.max().item() < 1e-4 and am == 1.0
    report={"passed":passed,"device":"cpu","dtype":"fp32","transformers":transformers.__version__,
            "torch":str(torch.__version__),"seed":seed,"config":vars(cfg),"input_shape":list(ids.shape),
            "params":n_ours,"transposed_weights":n_t,"max_abs_diff":d.max().item(),
            "argmax_match":am,"top5_set_overlap":top5_hit,
            "hf_state_dict_has_lm_head":"lm_head.weight" in hf.state_dict(),
            "hf_tied":hf.lm_head.weight is hf.transformer.wte.weight,
            "ours_tied":ours.head.weight is ours.wte.weight}
    if out_path is not None:
        target=Path(out_path);target.parent.mkdir(parents=True,exist_ok=True)
        payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
        with target.open("x") as f:
            f.write(payload)
    if not passed:
        raise AssertionError("HF对拍未通过，见报告")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="char")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--schedule-total", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", choices=["cpu","mps"], default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--parity-hf", action="store_true")
    ap.add_argument("--parity-small", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path(OUT_DIR))
    ap.add_argument("--data-path", type=Path, default=None)
    args = ap.parse_args()
    if args.parity_hf or args.parity_small:
        name = args.out_name or ("parity-small" if args.parity_small else "parity-hf")
        if Path(name).name != name or name in ("",".",".."):
            raise ValueError("out-name只接受文件名")
        target=args.out_dir/(name if name.endswith(".json") else name+".json")
        parity_hf(args.parity_small,args.seed,target)
    else:
        run_task(args)


if __name__ == "__main__":
    main()

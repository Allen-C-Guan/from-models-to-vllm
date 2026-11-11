# OpenWebText位置分片获取，供Book2第8/11章；已有语料和中断残件不覆盖。
# 在工作区激活env.sh后运行；--target-mb/--tok-mb沿用旧参数名，实际单位为百万Unicode字符。
# 缓存/产物在工作目录log；本程序不证明内容去重，不把字符预算称作UTF8字节大小。
import argparse
import hashlib
import json
from pathlib import Path

OUT_DIR=Path.cwd()/"log/book2-scaling"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target-mb",type=float,default=1150.,help="旧参数名：LM目标百万字符，不是MB")
    ap.add_argument("--tok-mb",type=float,default=30.,help="分词器分片的百万字符上限")
    ap.add_argument("--skip",type=int,default=6000,help="LM跳过的起始文档数；分词器只取其中一部分或全部")
    ap.add_argument("--revision",default=None,help="可固定数据集版本；不填则记录未固定")
    ap.add_argument("--out-dir",type=Path,default=OUT_DIR)
    a=ap.parse_args()
    if not 0<a.target_mb<float("inf") or not 0<a.tok_mb<float("inf") or a.skip<=0:
        raise ValueError("两个字符预算须有限正数，skip须正")
    tok=a.out_dir/"owt_tokenizer_corpus.txt";docs=a.out_dir/"owt_docs.jsonl";meta=a.out_dir/"subsample_meta.json"
    for path in (tok,docs,meta,Path(str(tok)+".part"),Path(str(docs)+".part")):
        if path.exists():raise FileExistsError(f"已有文件不覆盖，请另选out-dir：{path}")
    # 延迟导入，使--help和已有文件保护不依赖datasets。
    from datasets import load_dataset
    ds=load_dataset("Skylion007/openwebtext",split="train",streaming=True,revision=a.revision)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    strings=[];chars=0
    for i,example in enumerate(ds):
        if i>=a.skip:break
        text=example["text"];strings.append(text);chars+=len(text)
        if chars>=a.tok_mb*1e6:break
    if not strings:raise ValueError("没有分词器训练文本")
    tok_part=Path(str(tok)+".part");tok_part.write_text("\n\n".join(strings),encoding="utf-8")
    tok_count=len(strings);tok_chars=chars;del strings
    docs_part=Path(str(docs)+".part");count=chars=raw_bytes=0
    with docs_part.open("x",encoding="utf-8") as out:
        for i,example in enumerate(ds):
            if i<a.skip:continue
            text=example["text"]
            out.write(json.dumps({"text":text},ensure_ascii=False)+"\n")
            count+=1;chars+=len(text);raw_bytes+=len(text.encode("utf-8"))
            if chars>=a.target_mb*1e6:break
    if chars<a.target_mb*1e6:
        raise ValueError("来源耗尽，尚未达到目标字符量；保留.part现场，不标完成")
    tok_part.replace(tok);docs_part.replace(docs)
    report={"dataset":"Skylion007/openwebtext","requested_revision":a.revision,
            "tokenizer_source_positions":[0,tok_count],"lm_source_positions":[a.skip,a.skip+count],
            "tokenizer_documents":tok_count,"tokenizer_text_characters":tok_chars,
            "lm_documents":count,"lm_text_characters":chars,"lm_raw_utf8_bytes":raw_bytes,
            "lm_jsonl_bytes":docs.stat().st_size,"lm_jsonl_sha256":sha(docs),"tokenizer_text_sha256":sha(tok),
            "note":"位置分离不是内容去重；旧-mb参数按百万字符停止，文件字节另记"}
    with meta.open("x") as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

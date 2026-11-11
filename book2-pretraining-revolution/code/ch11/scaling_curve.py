# scaling_curve.py —— Book2第11章：汇编真实终点、计算量和锚点敏感性。
# 损失按SP/GPT-2词表分面；不再把JSONL字节和净文本字节混成同一个nat/byte轴。
# 运行：工作区激活env.sh后 python code/ch11/scaling_curve.py --help
# 只读取报告，不训练；历史表格汇总须显式--historical，输出到log且不覆盖已有结果。
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path.cwd()
CH11=ROOT/"log/book2-ch11"
CH08=ROOT/"log/book2-ch08"
FIG_DIR=CH11/"figures"
PREFIX="scaling_curve"
HISTORICAL=False
PF_DAY=8.64e19
GPT3_COMPUTE=3.1428e23  # 未圆整6×174.6B×300B，见第7/9章


def identity(path):
    return {"path":str(path),"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def complete_point(p,loss_key="eval_final"):
    """完整报告须有终步；出版表格汇总另以published_table明确标记，不当原始日志。"""
    value=p.get(loss_key)
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError("损失必须为有限非负数")
    if p.get("diverged") or p.get("completed") is False:
        raise ValueError("失败或未完成点不能入图")
    if p.get("source_kind")=="published_table":
        if not HISTORICAL:raise ValueError("出版表格汇总需显式--historical")
        return
    if p.get("steps",0)<=0 or p.get("steps")!=p.get("steps_planned"):
        raise ValueError("缺少完整训练步数")
    if not p.get("eval_losses") or p["eval_losses"][-1]["step"]!=p["steps"]:
        raise ValueError("缺少终步eval")


def build_points(iso,t124,fam):
    """各点保留原始nats/token，按词表标记；只统一6ND计算量，不猜评分字节分母。"""
    rows=[]
    def row(family,name,n,d,loss,source):
        if min(n,d)<=0 or not math.isfinite(n*d):
            raise ValueError("N/D须为有限正数")
        rows.append({"family":family,"name":name,"C":6.0*n*d,"N":n,"D":d,"L_tok":loss,
                     "tokenizer_group":"gpt2" if family=="gpt2_124m_fast" else "sp8k",
                     "source_kind":source.get("source_kind","run_report"),
                     "eval_identity":source.get("run_identity",source.get("run_config"))})
    for p in fam["points"]:
        complete_point(p)
        row("ch08_family",p["name"],p["N"],p["d_tokens_processed"],p["eval_final"],p)
    for p in iso["points"]:
        complete_point(p)
        row("iso_"+p["level"],p["name"],p["N"],p["D_tokens"],p["eval_final"],p)
        if not math.isclose(rows[-1]["C"],p["C_actual"],rel_tol=1e-10):
            raise ValueError("报告C_actual与完成的6ND不一致")
    run=t124["run"];complete_point(run)
    row("gpt2_124m_fast",f"124m@{run['D_tokens']/1e6:g}M",run["model"]["params_non_embedding"],
        run["D_tokens"],run["eval_final"],run)
    if len({(p["family"],p["name"]) for p in rows})!=len(rows):
        raise ValueError("重复的系列/点名")
    return rows


def optimal_analysis(iso,t124,rows):
    """候选外推及三种锚点选择的敏感性，不使用band作统计置信或最优性判断。"""
    sf=iso.get("slope_fit",{})
    views={k:sf[k]["slope"] for k in ("slope_argmin_all","slope_ingrid_vertices","slope_mixed")
           if sf.get(k) is not None}
    result={"gamma_views":views,
            "interpretation":"不同锚点的描述性斜率，不是上下界或置信区间",
            "literature_slopes":{"kaplan":.73,"chinchilla_isoflop":.50},
            "D_over_N_candidates":sf.get("D_over_N_candidates",{}),
            "L_star_power_in_C":sf.get("L_star_power_in_C_ingrid")}
    full=next(p for p in rows if p["tokenizer_group"]=="gpt2")
    high=next((f for f in iso.get("level_fits",[]) if f["level"]=="c3"),None)
    candidates=[]
    if high and high.get("N_star_inside_grid") and high.get("N_star",0)>0:
        for label,gamma in views.items():
            n=high["N_star"]*(full["C"]/high["C_actual_mean"])**gamma
            if math.isfinite(n) and n>0:
                candidates.append({"anchor_rule":label,"gamma":gamma,"N_candidate":n,
                                   "D_candidate":full["C"]/(6*n)})
    result["same_compute_extrapolation"]={"C":full["C"],"candidates":candidates,
        "note":"需要在同预算、同词表和同eval协议下新增实验验证，不能宣布实际模型偏离最优多少倍"}
    c2a=next((p for p in rows if p["name"]=="c2a"),None)
    f20=next((p for p in rows if p["family"]=="ch08_family" and p["name"]=="20m"),None)
    if c2a and f20:
        result["c2a_vs_20m"]={"compute_ratio":c2a["C"]/f20["C"],
                             "loss_difference":c2a["L_tok"]-f20["L_tok"],
                             "note":"只计算输入两点的差；解释优劣仍需核分词器和eval协议。出版历史记录声明二者同窗，非同预算或定律胜负实验"}
    return result


def convert_box(rows):
    iso=sum(p["C"] for p in rows if p["family"].startswith("iso_"))
    family=sum(p["C"] for p in rows if p["family"]=="ch08_family")
    full=next(p["C"] for p in rows if p["tokenizer_group"]=="gpt2")
    return {"pf_day":PF_DAY,"iso15_only_C":iso,"iso15_only_pf_days":iso/PF_DAY,
            "ch08_family_C":family,"ch08_family_pf_days":family/PF_DAY,
            "family_total_C":iso+family,"family_total_pf_days":(iso+family)/PF_DAY,
            "run124_C":full,"run124_pf_days":full/PF_DAY,"run124_vs_gpt3":full/GPT3_COMPUTE,
            "gpt3_reference_flops":GPT3_COMPUTE,
            "note":"6ND均按各自已声明N/D口径；两族合计不含另列的124M，非设备wall-clock"}


def configure_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei"):
        if name in names:
            plt.rcParams["font.family"]=name
            break
    plt.rcParams.update({"font.size":9,"axes.unicode_minus":False,"svg.fonttype":"none"})
    return plt


def save_figure(fig,slot):
    FIG_DIR.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","svg"):
        target=FIG_DIR/f"{PREFIX}-{slot}.{suffix}"
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=300,facecolor="white")


def make_fig_11_4(iso):
    plt=configure_plot()
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.7),dpi=300)
    for ax,level,color in zip(axes,("c1","c2","c3"),("#2a78d6","#1baf7a","#eb6834")):
        points=sorted([p for p in iso["points"] if p["level"]==level],key=lambda p:p["N"])
        for p in points:complete_point(p)
        ax.plot([p["N"] for p in points],[p["eval_final"] for p in points],"o-",color=color,lw=2)
        ax.set_xscale("log");ax.set_xlabel("非嵌入参数 N");ax.set_ylabel("验证损失（nat/token，SP）")
        ax.set_title(level.upper()+"：仅连接已测点")
        ax.grid(color="#e1e0d9",lw=.5)
    fig.suptitle("同预算下改变参数与数据：先看观测覆盖范围")
    fig.tight_layout()
    save_figure(fig,"fig-11-4");plt.close(fig)


def make_fig_11_5(rows):
    plt=configure_plot()
    fig,(sp,gpt,optimal)=plt.subplots(1,3,figsize=(12,4.5),dpi=300,
                                   gridspec_kw={"width_ratios":[1.25,.85,1.15]})
    family=sorted([p for p in rows if p["family"]=="ch08_family"],key=lambda p:p["C"])
    sp.plot([p["C"] for p in family],[p["L_tok"] for p in family],"x--",color="#2a78d6",
            label="第8章固定训练量族",lw=1.5,ms=6)
    best=[]
    for level,color in (("c1","#eb6834"),("c2","#1baf7a"),("c3","#eda100")):
        points=[p for p in rows if p["family"]=="iso_"+level]
        if not points:continue
        sp.scatter([p["C"] for p in points],[p["L_tok"] for p in points],color=color,label="iso "+level.upper())
        best.append(min(points,key=lambda p:p["L_tok"]))
    sp.set_xscale("log");sp.set_ylabel("验证损失（nat/token，SP-8k）")
    sp.set_title("SP词表：20个历史终点" if HISTORICAL else "SP词表：输入报告终点")
    sp.legend(fontsize=7.5,frameon=False)
    point=next(p for p in rows if p["tokenizer_group"]=="gpt2")
    gpt.scatter([point["C"]],[point["L_tok"]],marker="*",s=130,color="#52514e")
    gpt.set_xscale("log")
    gpt.set_xlim(point["C"]*.7,point["C"]/.7)
    gpt.set_ylim(point["L_tok"]-.3,point["L_tok"]+.3)
    gpt.annotate(f"124M / {point['D']/1e6:.3g}M token\n{point['L_tok']:.5f}",(point["C"],point["L_tok"]),
                 xytext=(0,15),textcoords="offset points",ha="center",fontsize=8)
    gpt.set_title("GPT-2词表：单独显示")
    gpt.set_ylabel("验证损失（nat/token，GPT-2）")
    best.sort(key=lambda p:p["C"])
    optimal.plot([p["C"] for p in best],[p["N"] for p in best],"o-",color="#2a78d6",
                 label="各档最佳已测点")
    if len(best)>1:
        anchor=best[len(best)//2];xs=np.geomspace(best[0]["C"],best[-1]["C"],60)
        for gamma,style,color in ((.73,"--","#eb6834"),(.5,":","#1baf7a")):
            optimal.plot(xs,anchor["N"]*(xs/anchor["C"])**gamma,style,color=color,
                         label=f"文献斜率{gamma}（平移截距）")
    optimal.set_xscale("log");optimal.set_yscale("log")
    optimal.set_title("参数位置：已测最佳并非连续最优")
    optimal.set_ylabel("非嵌入参数 N");optimal.legend(fontsize=7.3,frameon=False)
    for ax in (sp,gpt,optimal):
        ax.set_xlabel("计算量 C≈6ND（FLOPs）");ax.grid(color="#e1e0d9",lw=.5);ax.set_axisbelow(True)
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter
    sp.xaxis.set_major_locator(FixedLocator([3e14,1e15,3e15,6e15]))
    sp.xaxis.set_major_formatter(FixedFormatter([r"$3\times10^{14}$",r"$10^{15}$",r"$3\times10^{15}$",r"$6\times10^{15}$"]))
    gpt.xaxis.set_major_locator(FixedLocator([2e16,3e16]))
    gpt.xaxis.set_major_formatter(FixedFormatter([r"$2\times10^{16}$",r"$3\times10^{16}$"]))
    optimal.xaxis.set_major_locator(FixedLocator([6e14,2e15,6e15]))
    optimal.xaxis.set_major_formatter(FixedFormatter([r"$6\times10^{14}$",r"$2\times10^{15}$",r"$6\times10^{15}$"]))
    optimal.yaxis.set_major_locator(FixedLocator([.6e6,2e6,6e6]))
    optimal.yaxis.set_major_formatter(FixedFormatter(["0.6M","2M","6M"]))
    for ax in (sp,gpt,optimal):
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x",labelsize=8)
    optimal.yaxis.set_minor_formatter(NullFormatter())
    note="历史表格终点，未重跑训练。" if HISTORICAL else "输入报告终点；非设备性能基准。"
    fig.text(.02,.015,note+" 损失按词表分面；原JSONL字节与净文本字节系数不能混成共同纵轴。",fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1))
    save_figure(fig,"fig-11-5");plt.close(fig)


def main():
    global FIG_DIR,PREFIX,HISTORICAL
    ap=argparse.ArgumentParser()
    ap.add_argument("--iso-report",type=Path,default=CH11/"iso_flop.json")
    ap.add_argument("--run-report",type=Path,default=CH11/"train124_fast.json")
    ap.add_argument("--family-report",type=Path,default=CH08/"family_fast.json")
    ap.add_argument("--out-dir",type=Path,default=CH11/"scaling")
    ap.add_argument("--out-name",default="scaling_curve")
    ap.add_argument("--historical",action="store_true",help="输入为出版历史表格汇总，非恢复的原始日志")
    a=ap.parse_args()
    PREFIX=a.out_name
    if Path(PREFIX).name!=PREFIX or PREFIX in ("",".",".."):raise ValueError("out-name只接受名称")
    HISTORICAL=a.historical;FIG_DIR=a.out_dir/"figures"
    files=[a.out_dir/f"{PREFIX}.json",a.out_dir/f"{PREFIX}-points.csv"]
    files += [FIG_DIR/f"{PREFIX}-fig-11-{i}.{ext}" for i in (4,5) for ext in ("png","svg")]
    for path in files:
        if path.exists():raise FileExistsError(f"不覆盖已有产物：{path}")
    iso=json.loads(a.iso_report.read_text());full=json.loads(a.run_report.read_text());family=json.loads(a.family_report.read_text())
    rows=build_points(iso,full,family);analysis=optimal_analysis(iso,full,rows);box=convert_box(rows)
    report={"meta":{"sources":[identity(p) for p in (a.iso_report,a.run_report,a.family_report)],
                    "historical_table_summary":HISTORICAL,
                    "loss_units":"分别为各词表的nat/token，无共同byte分母"},"rows":rows,
            "optimal":analysis,"convert_box":box}
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    a.out_dir.mkdir(parents=True,exist_ok=True)
    make_fig_11_4(iso);make_fig_11_5(rows)
    with files[0].open("x") as f:f.write(payload)
    with files[1].open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["family","name","tokenizer_group","C","N","D","L_tok"],extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"points":len(rows),"gamma_views":analysis["gamma_views"],"compute":box},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

# Book3第7章：实验图只读完整报告；机制图按论文r斜坡，明确区别HF索引斜坡。
# python make_figures.py --only 4 --out-dir log/book3-ch07/figures
# 原passkey全零矩阵可用--historical-passkey-summary重画；仅转绘原章明确记载的计数。
from pathlib import Path
import argparse,json,math,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BLUE,ORANGE,TEAL,YELLOW='#2a78d6','#eb6834','#1baf7a','#eda100'
COLORS={'none':BLUE,'pi':ORANGE,'ntk':TEAL,'yarn_lite':YELLOW}
LABELS={'none':'none','pi':'PI','ntk':'static NTK','yarn_lite':'yarn_lite'}
plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':9,'axes.grid':True,'grid.color':'#e1e0d9','grid.linewidth':.5,'axes.axisbelow':True})


def save(fig,folder,slug,svg=False):
    fig.tight_layout();fig.savefig(folder/(slug+'.png'),facecolor='white')
    if svg:fig.savefig(folder/(slug+'.svg'),facecolor='white')
    plt.close(fig)


def load_extrapolate(path,historical=False):
    r=json.loads(Path(path).read_text());train=r.get('train',{})
    if r.get('completed') is False or train.get('completed') is False or train.get('failure'):raise ValueError('实验失败或未完成')
    if r.get('completed') is not True and not historical:raise ValueError('旧报告须显式historical')
    if not r.get('meta',{}).get('eval_only') and (not train.get('steps_done') or train['steps_done']!=train.get('steps_planned')):raise ValueError('训练步数未完成')
    ev=r['intervention_eval'];lengths=sorted(int(n) for n in ev['none'])
    if set(ev)!=set(COLORS) or not lengths:raise ValueError('四干预缺失')
    for iv,values in ev.items():
        if sorted(int(n) for n in values)!=lengths:raise ValueError('干预长度集不同')
        for n,entry in values.items():
            if not math.isfinite(entry['loss']) or entry['n_tokens']<=0:raise ValueError('无效评估结果')
            if not math.isclose(math.exp(entry['loss']),entry['ppl'],rel_tol=2e-4,abs_tol=.002):raise ValueError('loss/PPL不匹配')
            parse_buckets(entry['buckets'],int(n))
    return r,lengths


def parse_buckets(buckets,n):
    values=[]
    for name,value in buckets.items():
        match=re.fullmatch(r'pos\[(\d+):(\d+)\]',name)
        if not match or not math.isfinite(value):raise ValueError('分桶键或数值无效')
        lo,hi=map(int,match.groups());values.append((lo,hi,value))
    values.sort()
    if not values or values[0][0]!=0 or values[-1][1]!=n:raise ValueError('桶未覆盖完整窗口')
    if any(lo>=hi for lo,hi,_ in values) or any(a[1]!=b[0] for a,b in zip(values,values[1:])):raise ValueError('桶重叠或有缺口')
    return values


def merge_buckets(buckets,target):
    source=parse_buckets(buckets,target[-1]);result=[]
    for lo,hi in zip(target,target[1:]):
        selected=[(a,b,v) for a,b,v in source if lo<=a and b<=hi]
        if not selected or sum(b-a for a,b,_ in selected)!=hi-lo:raise ValueError('目标边界不能切开原桶或遗漏位置')
        result.append(sum((b-a)*v for a,b,v in selected)/(hi-lo))
    return result


def load_passkey(path,historical=False):
    r=json.loads(Path(path).read_text())
    if r.get('completed') is False or (r.get('completed') is not True and not historical):raise ValueError('旧报告须historical；明确失败不可绘图')
    meta=r['meta'];lengths=meta['lengths'];depths=meta['depths'];trials=meta['trials'];counts={}
    if not lengths or not depths or trials<=0:raise ValueError('空实验协议')
    for iv in COLORS:
        cells=r['results'][iv];counts[iv]={}
        for length in lengths:
            expected=[f'len{length}@d{float(d)}' for d in depths]
            # 旧Python浮点格式与此相同；不把缺失设置补成零。
            if any(name not in cells for name in expected):raise ValueError('缺失passkey设置格')
            selected=[cells[name] for name in expected]
            for cell in selected:
                if cell['trials']!=trials or not 0<=cell['hits']<=trials:raise ValueError('无效计数')
                if not math.isclose(cell['accuracy'],cell['hits']/trials,abs_tol=1e-4):raise ValueError('计数与准确率不符')
            total=sum(x['trials'] for x in selected);hits=sum(x['hits'] for x in selected)
            counts[iv][str(length)]=dict(hits=hits,trials=total)
    return r,counts


def fig_passkey(counts,folder,historical=False):
    lengths=sorted(int(x) for x in counts['none']);names=list(COLORS)
    fig,ax=plt.subplots(figsize=(6.4,3.8));ax.grid(False)
    matrix=np.array([[counts[iv][str(n)]['hits']/counts[iv][str(n)]['trials'] for n in lengths] for iv in names])
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i,iv in enumerate(names):
        for j,n in enumerate(lengths):
            rec=counts[iv][str(n)];ax.text(j,i,f"{rec['hits']}/{rec['trials']}",ha='center',va='center',color='#0b0b0b')
    ax.set_xticks(range(len(lengths)),[str(n) for n in lengths]);ax.set_yticks(range(4),[LABELS[n] for n in names]);ax.set_xlabel('prompt length (tokens)')
    ax.set_title('Passkey hits / trials (depths pooled)')
    fig.text(.5,.015,'Historical chapter counts; no new model evaluation' if historical else 'Same contexts paired across interventions; substring matching',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(folder/'fig-7-3-passkey-accuracy.png',facecolor='white');plt.close(fig)


def mechanism_arrays(D=64,L=2048,scale=4,base=10000):
    i=np.arange(D//2);theta=base**(-2*i/D);lam=2*np.pi/theta;r=L/lam
    return i,lam,np.full(D//2,1/scale),scale**(-2*i/(D-2)),(1-np.clip((r-1)/31,0,1))/scale+np.clip((r-1)/31,0,1)


def fig_mechanism(folder):
    i,lam,pi,ntk,yarn=mechanism_arrays();fig,ax=plt.subplots(figsize=(6.4,4))
    for y,color,label in [(pi,ORANGE,'PI: uniform 1/s'),(ntk,TEAL,'static NTK: base change'),(yarn,YELLOW,'YaRN paper: ramp in r')]:ax.plot(i,y,c=color,lw=2,label=label)
    slow=int(np.flatnonzero(lam>=2048)[0]);ax.axvspan(slow-.5,31.5,color='#f0efec',zorder=0)
    ax.text(25,.58,'less than one cycle\nin a 2048-position window',ha='center',fontsize=8,color='#52514e')
    ax.set_yscale('log');ax.set_yticks([.25,.5,1],['1/s','.5','1']);ax.set_xlabel('frequency-pair index i (fast to slow)');ax.set_ylabel('new / original inverse frequency')
    ax.set_title('RoPE frequency changes (D=64, L=2048, s=4)');ax.legend(loc='center left',frameon=False,fontsize=8)
    save(fig,folder,'fig-7-4-intervention-mechanism',True)


def fig_experiments(report,lengths,folder,only):
    ev=report['intervention_eval'];model=report['meta']['model'];train_window=model['max_position_embeddings']
    if '1' in only:
        fig,ax=plt.subplots(figsize=(6.4,4));ys=[ev['none'][str(n)]['loss'] for n in lengths]
        ax.plot(lengths,ys,'o-',c=BLUE,lw=2);ax.set_xlabel('evaluation window');ax.set_ylabel('eval loss (nat)');ax.set_title('Direct extrapolation: actual report')
        # 600步历史先例不自动混进任何新报告。
        save(fig,folder,'fig-7-1-direct-extrapolation-ppl')
    if '2' in only:
        fig,ax=plt.subplots(figsize=(6.4,4))
        for n in lengths:
            values=parse_buckets(ev['none'][str(n)]['buckets'],n)
            ax.plot([hi for _,hi,_ in values],[v for _,_,v in values],marker='o',lw=2,label=f'window {n}')
        ax.axvline(train_window,color='#898781',ls=':');ax.set_xlabel('position bucket upper boundary');ax.set_ylabel('bucket loss');ax.legend(frameon=False);save(fig,folder,'fig-7-2-position-buckets')
    if '5' in only:
        fig,ax=plt.subplots(figsize=(6.4,4))
        for iv in COLORS:ax.plot(lengths,[ev[iv][str(n)]['loss'] for n in lengths],marker='o',lw=2,c=COLORS[iv],label=LABELS[iv])
        ax.axvline(train_window,color='#898781',ls=':');ax.set_xlabel('evaluation window');ax.set_ylabel('eval loss (nat)');ax.set_title('Four interventions, one checkpoint');ax.legend(frameon=False);save(fig,folder,'fig-7-5-four-interventions')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='4');ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book3-ch07/figures')
    ap.add_argument('--extrapolate-report',type=Path);ap.add_argument('--passkey-report',type=Path);ap.add_argument('--historical',action='store_true')
    ap.add_argument('--historical-passkey-summary',action='store_true',help='只转绘原章已刊四干预×三长度×三深度×8次全零计数')
    args=ap.parse_args();only=set(args.only.split(','))
    if not only or not only<=set('12345'):raise ValueError('only可选1..5')
    if args.historical_passkey_summary and (only!={'3'} or args.passkey_report):raise ValueError('历史计数转绘只允许单独图3')
    report=lengths=counts=None
    if only&set('125'):
        if not args.extrapolate_report:raise ValueError('实验图必须提供完整报告，不反造缺失曲线')
        report,lengths=load_extrapolate(args.extrapolate_report,args.historical)
    if '3' in only:
        if args.historical_passkey_summary:
            counts={iv:{str(n):dict(hits=0,trials=24) for n in [1536,4096,8192]} for iv in COLORS}
        elif args.passkey_report:_,counts=load_passkey(args.passkey_report,args.historical)
        else:raise ValueError('图3需要完整passkey报告或明确历史计数转绘')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for number in only:
        if list(args.out_dir.glob(f'fig-7-{number}-*')):raise FileExistsError('输出图存在，请用新目录')
    if '4' in only:fig_mechanism(args.out_dir)
    if '3' in only:fig_passkey(counts,args.out_dir,args.historical_passkey_summary)
    if report is not None:fig_experiments(report,lengths,args.out_dir,only)


if __name__=='__main__':main()

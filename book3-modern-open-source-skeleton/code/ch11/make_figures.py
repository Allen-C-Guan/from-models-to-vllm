# Book3第11章：丛书阅读关系与实际代码资产。只绘示意，不读取设计文档正文。
# python make_figures.py --out-dir log/book3-ch11/figures
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE,ORANGE,TEAL,YELLOW='#2a78d6','#eb6834','#1baf7a','#eda100'
MID='#52514e'
plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':9})
def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,color='#0b0b0b')
def arrow(ax,a,b,color=MID,style='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.5,linestyle=style))
def finish(fig,path):
    if path.exists():raise FileExistsError(path)
    fig.tight_layout();fig.savefig(path,facecolor='white');plt.close(fig)
def fig_11_1(folder):
    fig,ax=plt.subplots(figsize=(9,5));ax.set(xlim=(0,12),ylim=(0,10));ax.axis('off')
    rows=[('Book1 -> Book2','Transformer computation -> pretraining objectives and GPT-2',BLUE),
          ('Book3: current volume','RMSNorm / SwiGLU / RoPE / GQA -> integrated LLaMA-style model',ORANGE),
          ('Book4','Conditional computation, low-rank state, numerical representation',BLUE),
          ('Book5','Sparse reads, recurrent state and hybrid architectures',BLUE),
          ('Book6 -> Book7 -> Book8','Inference principles -> vLLM V1 architecture -> source code',TEAL)]
    for i,(title,body,color) in enumerate(rows):
        y=8.25-1.6*i;box(ax,.3,y,2.4,1.,title,color);box(ax,3.4,y,8.2,1.,body,color)
        arrow(ax,(2.7,y+.5),(3.4,y+.5))
    ax.text(6,9.65,'From model components to inference systems',ha='center',fontsize=12)
    ax.text(6,.45,'Reading relationships; later practice does not require completing the Book3 full training budget.',ha='center',fontsize=8,color=MID)
    finish(fig,folder/'fig-11-1-eight-book-map.png')
def inventory(book):
    core=sorted(p for p in (book/'code').glob('ch*/*.py') if not p.name.startswith('make_fig'))
    refs=sorted((book/'code/00-feasibility').glob('*.py'))
    return core,refs
def fig_11_2(folder,book):
    required=['ch02/rmsnorm.py','ch03/swiglu.py','ch04/rope.py','ch06/gqa.py','ch10/llama215.py',
              'ch10/train_215.py','ch10/prep_sp32k.py','00-feasibility/llama_slots.py']
    if not all((book/'code'/f).is_file() for f in required):raise ValueError('缺组件文件，不能画成完整依赖图')
    core,refs=inventory(book)
    fig,ax=plt.subplots(figsize=(10,6.7));ax.set(xlim=(0,15),ylim=(0,12));ax.axis('off')
    ax.text(7.5,11.65,'Code assets: imports, experiments and supporting checks',ha='center',fontsize=12)
    box(ax,.3,9.4,4.1,1.25,'Book2 ch10/model.py\nGPT-2 used by module experiments',MID)
    box(ax,.3,7.2,4.1,1.35,'ch02 / ch03 / ch04 / ch05 / ch06\nindependent module experiments\n(not successive trained checkpoints)',BLUE)
    arrow(ax,(2.35,9.4),(2.35,8.55))
    box(ax,5.1,6.6,4.4,3.3,'Four teaching components\nch02/rmsnorm.py\nch03/swiglu.py\nch04/rope.py\nch06/gqa.py',ORANGE)
    box(ax,10.3,8.5,4.4,1.4,'ch10/llama215.py\n207,119,360 parameters\nimports these four components',ORANGE)
    arrow(ax,(9.5,9.1),(10.3,9.1),ORANGE)
    box(ax,10.3,5.7,4.4,1.35,'ch10/train_215.py\ntraining, evaluation, checkpoints',YELLOW)
    arrow(ax,(12.5,8.5),(12.5,7.05),ORANGE)
    box(ax,10.3,2.8,4.4,1.35,'ch10/prep_sp32k.py\ntokenizer + token streams',YELLOW)
    arrow(ax,(12.5,4.15),(12.5,5.7),YELLOW)
    ax.text(12.85,4.9,'data',fontsize=8,color=MID)
    box(ax,5.1,3.4,4.4,1.6,'00-feasibility/llama_slots.py\ncomponent reference\nLLaMAConfig for chapter10',TEAL)
    arrow(ax,(7.3,5.),(7.3,6.6),TEAL,'--')
    ax.text(7.65,5.65,'numerical comparison',fontsize=7.2,color=MID)
    box(ax,.3,3.5,4.1,2.3,'Other reading / verification tools\nch01: handcalc\nch05: config_infer\nch07: extrapolate + passkey\nch09: config_reverse',BLUE)
    box(ax,5.1,1.,9.6,1.,'Companion guide: code/ch10/DESIGN-215M.md\nCheck the current environment and command interfaces before restarting.',MID)
    ax.text(7.5,.3,f'{len(core)} core implementation / experiment files; {len(refs)} feasibility/reference files; figure scripts counted separately.',ha='center',fontsize=8,color=MID)
    finish(fig,folder/'fig-11-2-code-assets.png')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,default=Path('log/book3-ch11/figures'))
    ap.add_argument('--only',choices=['1','2','1,2'],default='1,2')
    ap.add_argument('--book-dir',type=Path,default=Path(__file__).resolve().parents[2]);a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    if '1' in a.only:fig_11_1(a.out_dir)
    if '2' in a.only:fig_11_2(a.out_dir,a.book_dir)
if __name__=='__main__':main()

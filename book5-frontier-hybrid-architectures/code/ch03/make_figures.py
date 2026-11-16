# Book5 第3章：NSA/DSA学习路径对照与旧版本gap记录。
# source env.sh && python code/ch03/make_figures.py --out-dir log/book5-ch03/figures-run1
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
FONT=FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":FONT.get_name(),"axes.unicode_minus":False,"font.size":11})

def comparisons(out):
    fig,ax=plt.subplots(figsize=(12,8));ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.25,.96,"NSA：三分支联合训练",ha="center",fontsize=16,color="#2678c8")
    ax.text(.75,.96,"DSA：独立索引器与主注意力",ha="center",fontsize=16,color="#ce6732")
    pairs=[
      ("压缩分支读取摘要\n输出参与主任务损失，因而获得梯度","索引器扫描历史索引键\n输出各位置的索引分 I"),
      ("复用压缩注意力权重\n网格聚合，再在 GQA 组内求和","KL 对齐主注意力分布\n老师停止梯度，索引器输入 detach"),
      ("top-n 选块，读取原始 token\n同组查询头共享所选块","top-k 选 token\n全部查询头共享所选潜记录"),
      ("另设局部窗口分支\n三分支输出经 sigmoid 门加权","选中集内执行 MLA 主注意力\n主模型由语言建模损失更新"),
      ("从零预训练 270B token\n之后做长上下文适配","从 V3.1-Terminus 续训\n热身约 2.1B，稀疏阶段约 943.7B"),
      ("原始记录仍需保存，另有摘要\n表4算的是等效读取量","主缓存之外还有索引键\n索引器全长扫描仍有成本")]
    for i,(l,r) in enumerate(pairs):
        y=.80-i*.135
        for x,text,color in [(.03,l,"#2678c8"),(.53,r,"#ce6732")]:
            ax.add_patch(FancyBboxPatch((x,y),.44,.10,boxstyle="round,pad=.01",fc="#f7f9fb",ec=color))
            ax.text(x+.22,y+.05,text,ha="center",va="center",fontsize=11)
    ax.text(.5,.035,"共同点：内容相关选择；离散 top-k 本身不提供普通反向梯度。",ha="center",fontsize=11)
    fig.tight_layout();fig.savefig(out/"fig-3-2-nsa-to-dsa-subtraction.png",dpi=170);plt.close(fig)

def history(out):
    fig,ax=plt.subplots(figsize=(9,5.3))
    x=[4,8,12,16];y=[.564,.249,.0998,3e-7]
    ax.plot(x,y,"o-",color="#2678c8")
    ax.set(yscale="log",xticks=x,ylim=(1e-8,2),xlabel="每查询块选择的块数（共16块）",
           ylabel="相对 L2 输出差",title="旧前视版本的历史记录：不作为当前因果实现的质量证据")
    for xx,yy,label in zip(x,y,["0.564","0.249","0.0998",r"约 $3.0\times10^{-7}$"]):
        ax.annotate(label,(xx,yy),xytext=(-10,10),textcoords="offset points",ha="right" if xx==16 else "center")
    ax.text(.04,.08,"CPU FP32；seed 20261002；n=1024，块宽64\n选全块可通过对拍，仍不足以发现稀疏选择中的前视。",transform=ax.transAxes,fontsize=10)
    ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/"fig-3-3-block-sparse-gap-ladder.png",dpi=170);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False);comparisons(a.out_dir);history(a.out_dir)
if __name__=="__main__":main()

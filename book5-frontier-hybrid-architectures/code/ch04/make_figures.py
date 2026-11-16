# Book5 第4章：V4整机与层内读取、GLM-5.2索引共享排班。
# 由 dsv4_glm5_reverse.py 的 --plot-dsv4-map / --plot-layer-map 调用，输出到独立log目录。
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,Patch
FONT=FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":FONT.get_name(),"axes.unicode_minus":False,"font.size":11})
BLUE="#2678c8";ORANGE="#ce6732"
def box(ax,x,y,w,h,text,color=BLUE,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.008",fc="#f6f8fb",ec=color,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs)
def arrow(ax,a,b):
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.3,color="#555"))
def draw_dsv4(path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    fig,ax=plt.subplots(figsize=(13,8));ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.5,.97,"DSV4-Flash：先保存不同粒度的历史，再统一读取",ha="center",fontsize=16)
    left=[
      (.77,"词嵌入，展开4条残差通道\n(B,n) → (B,n,4,4096)"),
      (.50,"重复43个主干块\n每块：mHC包住注意力，mHC包住MoE\n前2块纯滑窗；其余21 CSA + 20 HCA\n残差为(B,n,4,4096)\n各子层输入/输出仍为(B,n,4096)"),
      (.23,"汇合残差、归一化、输出头\n(B,n,4,4096) → (B,n,4096)\n→ (B,n,129280)")]
    for y,t in left:box(ax,.02,y,.35,.18,t)
    for a,b in zip(left,left[1:]):arrow(ax,(.195,a[0]),(.195,b[0]+.18))
    ax.text(.195,.10,"MTP是另计的训练模块\n不混入43层主干计数",ha="center",fontsize=10)
    rows=[
      (.79,"本层输入 H：(B,n,4096)\n分别投影查询、局部KV与压缩内容/权重"),
      (.63,"CSA：步长4、双路重叠聚合，索引选摘要\nHCA：每128个位置聚合，读取全部可见摘要\n摘要为 (B,Nc,512)，Nc=已闭合块数"),
      (.47,"选中/可见摘要 + 最近 r≤128 条局部KV\n沿记录维拼接：(B,M+r,512)，K=V一份\nquery：(B,64,1,512)"),
      (.31,"一次缩放点积 → 加掩码与sink → softmax → 值加权\n输出 (B,64,1,512)，再按查询位置作逆旋转\n最后64维旋转；不恢复摘要内原始token"),
      (.15,"分组低秩输出：8组，每组4096→1024\n拼接8192→4096，回到 (B,1,4096)")]
    for y,t in rows:box(ax,.43,y,.55,.125,t,ORANGE,10)
    for a,b in zip(rows,rows[1:]):arrow(ax,(.705,a[0]),(.705,b[0]+.125))
    ax.text(.705,.055,"位置配置按层型绑定：纯滑窗层用main；CSA/HCA整层用compress\n层内局部记录与摘要不各用一套频率。B为批量，M为实际读取的摘要数。",ha="center",fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def draw_glm(path,raw):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    types=raw["indexer_types"];assert len(types)==78 and types.count("full")==21
    fig,(ax,zoom)=plt.subplots(2,1,figsize=(12,5.4),gridspec_kw={"height_ratios":[1,1.2]})
    for i,t in enumerate(types):
        ax.barh(0,.92,left=i,height=.5,color=BLUE if t=="full" else "#f2d9c9")
    ax.set(xlim=(-1,79),ylim=(-.5,.9),yticks=[],xlabel="主干层号（从0开始）",
           title="GLM-5.2：21个full索引器，57个shared层；MTP另计")
    ax.set_xticks([0,2,6,10,20,30,40,50,60,70,77])
    ax.legend(handles=[Patch(color=BLUE,label="F：本层评分"),Patch(color="#f2d9c9",label="S：复用最近F的索引")],
              loc="upper center",ncol=2,frameon=False)
    for i,t in enumerate(types[:12]):
        zoom.barh(0,.92,left=i,height=.5,color=BLUE if t=="full" else "#f2d9c9")
        zoom.text(i+.46,0,t[0].upper(),ha="center",va="center",color="white" if t=="full" else "#85491f")
    zoom.set(xlim=(-.2,12),ylim=(-1.05,.5),yticks=[],xticks=range(12))
    zoom.text(6,-.65,"共享的是当前查询的索引，形状为 (B,nq,k)\n每层仍用自己的query和KV计算注意力；下个F层更新索引。",ha="center",fontsize=11)
    for axes in [ax,zoom]:
        for side in ["top","left","right"]:axes.spines[side].set_visible(False)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

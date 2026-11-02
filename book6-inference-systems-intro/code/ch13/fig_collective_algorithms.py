"""第13章：通信算法的逐轮机制图；独立构造示意，不是硬件连接或性能实测。

source ./env.sh
python code/Book6-推理系统导论/ch13/fig_collective_algorithms.py
图中Ring数值直接取自collective_walkthrough.py已核验的状态。
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collective_walkthrough import build_examples

INK = '#0b0b0b'
MUTED = '#52514e'
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
NAMES = 'ABCD'


def setup(ax, title):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.set_title(title, fontsize=12, color=INK, pad=10)


def box(ax, x, y, label, w=.25, h=.16):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle='round,pad=0.008', facecolor='#f3f6fa',
                 edgecolor='#8999ac', linewidth=1.1, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=9, color=INK, zorder=4)


def edge(ax, src, dst, label='', color=COLORS[0], offset=(0, 0), both=False, label_at=.5):
    ax.annotate('', xy=dst, xytext=src, arrowprops={
        'arrowstyle': '<->' if both else '->', 'lw': 1.7, 'color': color,
        'shrinkA': 4, 'shrinkB': 4}, zorder=2)
    if label:
        ax.text(src[0]*(1-label_at)+dst[0]*label_at+offset[0],
                src[1]*(1-label_at)+dst[1]*label_at+offset[1],
                label, ha='center', va='center', fontsize=8.5, color=INK,
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1}, zorder=5)


def save(fig, out, name):
    fig.tight_layout(pad=1.8)
    fig.savefig(out/name, dpi=300, facecolor='white')
    plt.close(fig)


def trees(out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ('Broadcast: same X (m,) along every edge',
         ['r0: X', 'r1: X', 'r2: X', 'r3: X'],
         [(0, 2, '1: X (m,)'), (0, 1, '2: X (m,)'), (2, 3, '2: X (m,)')]),
        ('Scatter: forward only the destination bundle',
         ['r0: A', 'r1: B', 'r2: C', 'r3: D'],
         [(0, 2, '1: CD (2m,)'), (0, 1, '2: B (m,)'), (2, 3, '2: D (m,)')]),
        ('Gather: concatenate; no elementwise sum',
         ['r0: ABCD', 'r1: B', 'r2: CD', 'r3: D'],
         [(1, 0, '1: B (m,)'), (3, 2, '1: D (m,)'), (2, 0, '2: CD (2m,)')]),
        ('Reduce: add corresponding elements',
         ['r0: x0+x1+x2+x3', 'r1: x1', 'r2: x2+x3', 'r3: x3'],
         [(1, 0, '1: x1 (m,)'), (3, 2, '1: x3 (m,)'), (2, 0, '2: x2+x3 (m,)')])]
    coords = [(.2, .74), (.2, .25), (.78, .74), (.78, .25)]
    for ax, (title, labels, edges) in zip(axes.flat, panels):
        setup(ax, title)
        for r, (x, y) in enumerate(coords):
            box(ax, x, y, labels[r], w=.36 if r == 0 else .30)
        for src, dst, label in edges:
            a, b = coords[src], coords[dst]
            if a[0] == b[0]:
                a = (a[0], a[1] + (-.1 if b[1]<a[1] else .1))
                b = (b[0], b[1] + (.1 if b[1]<a[1] else -.1))
                offset=(.10, 0)
            else:
                a = (a[0]+(.17 if b[0]>a[0] else -.17), a[1])
                b = (b[0]+(-.17 if b[0]>a[0] else .17), b[1])
                offset=(0, .05)
            edge(ax, a, b, label, offset=offset)
        ax.text(.5, .025, '1 / 2: logical round; arrows show payload shape',
                ha='center', fontsize=8, color=MUTED)
    save(fig, out, 'fig-13-4-collective-trees.png')


def ring_panels(out, data, phase, number):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    coords = [(.20, .72), (.80, .72), (.80, .25), (.20, .25)]
    starts = [(.36,.72),(.80,.62),(.64,.25),(.20,.35)]
    ends = [(.64,.72),(.80,.35),(.36,.25),(.20,.62)]
    offsets = [(0,.065),(.12,0),(0,-.07),(-.12,0)]
    for ax, step in zip(axes, data[phase]):
        setup(ax, f'Round {step["round"]}: '+('receive + local sum' if phase=='reduce_scatter' else 'receive + place'))
        for r, (x,y) in enumerate(coords):
            if phase == 'reduce_scatter':
                label = f'r{r}\n{NAMES[step["chunks_after"][r]]}: {step["values_after"][r]}'
            else:
                label = f'r{r}\n'+''.join(NAMES[j] for j in step['chunks_after'][r])
            box(ax, x, y, label, w=.27, h=.18)
        for msg in step['messages']:
            r=msg['src']
            label=f'{NAMES[msg["chunk"]]}: {msg["value"]}'
            edge(ax, starts[r], ends[r], label, COLORS[msg['chunk']], offsets[r])
        ax.text(.5, .02, 'Each arrow: one chunk (1,); boxes: state after round',
                ha='center', fontsize=8, color=MUTED)
    save(fig, out, f'fig-13-{number}-ring-{phase.replace("_", "-")}.png')


def recursive(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    xs = [.14, .38, .62, .86]
    ys = [.82, .49, .16]
    titles=['Recursive halving: keep only owned destinations',
            'Recursive doubling: exchange the full partial sum']
    rows=[
        [['ABCD\n(4,)' for _ in xs],
         ['AB: 101,202','AB: 1010,2020','CD: 303,404','CD: 3030,4040'],
         ['A:1111','B:2222','C:3333','D:4444']],
        [[f'x{r}\n(4,)' for r in range(4)],
         ['x0+x1','x0+x1','x2+x3','x2+x3'],
         ['sum(x0..x3)' for _ in xs]]]
    peers=[[2,1],[1,2]]
    for ax, title, states, distances in zip(axes,titles,rows,peers):
        setup(ax,title)
        for step,row in enumerate(states):
            for r,label in enumerate(row):
                box(ax,xs[r],ys[step],f'r{r}\n{label}',w=.205,h=.15)
        for step,distance in enumerate(distances):
            for r in range(4):
                peer=r^distance
                edge(ax,(xs[r],ys[step]-.085),(xs[peer],ys[step+1]+.085),
                     color=COLORS[r])
            size='(2,)' if step==0 else '(1,)'
            label=f'Round {step+1}: XOR {distance}; payload '+(size if ax is axes[0] else '(4,)')
            ax.text(.5,(ys[step]+ys[step+1])/2,label,ha='center',fontsize=9,
                    bbox={'facecolor':'white','edgecolor':'none','pad':2},zorder=5)
        ax.text(.5,.015,'Arrows: transmitted values; each receiver also uses local values',
                ha='center',fontsize=8,color=MUTED)
    save(fig,out,'fig-13-7-recursive-exchange.png')


def alltoall(out):
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    ys=[.79,.59,.39,.19]
    for k,ax in enumerate(axes,1):
        setup(ax,f'Round {k}: destination = (rank + {k}) mod 4')
        for r,y in enumerate(ys):
            box(ax,.12,y,f'r{r}',w=.17,h=.11)
            box(ax,.88,y,f'r{r}',w=.17,h=.11)
            dst=(r+k)%4
            edge(ax,(.23,y),(.77,ys[dst]),f'a{r}{dst} (m,)',COLORS[r],
                 label_at=.12)
        ax.text(.12,.96,'sender',ha='center',fontsize=9)
        ax.text(.88,.96,'receiver',ha='center',fontsize=9)
        ax.text(.5,.025,'Self block a_rr stays local; no elementwise sum',
                ha='center',fontsize=8,color=MUTED)
    save(fig,out,'fig-13-8-alltoall-pairwise.png')


def hierarchy(out):
    fig,axes=plt.subplots(1,3,figsize=(12,4.7))
    coords=[(.18,.71),(.18,.29),(.82,.71),(.82,.29)]
    titles=['1. Local Reduce-Scatter','2. Cross-node All-Reduce','3. Local All-Gather']
    states=[['A01','B01','A23','B23'],['A','B','A','B'],['AB','AB','AB','AB']]
    for stage,(ax,title,labels) in enumerate(zip(axes,titles,states)):
        setup(ax,title)
        ax.axvline(.5,color='#c4c8ce',ls='--',lw=1)
        ax.text(.18,.94,'node 0',ha='center',fontsize=10)
        ax.text(.82,.94,'node 1',ha='center',fontsize=10)
        for r,(x,y) in enumerate(coords):
            box(ax,x,y,f'r{r}: {labels[r]}',w=.30,h=.16)
        if stage==1:
            edge(ax,(.35,.71),(.65,.71),'A (M/2 bytes)',both=True,offset=(0,.07))
            edge(ax,(.35,.29),(.65,.29),'B (M/2 bytes)',both=True,color=COLORS[1],offset=(0,.07))
        else:
            for x in [.18,.82]:
                edge(ax,(x,.60),(x,.40),'(M/2 bytes)',both=True,offset=(.13,0))
        ax.text(.5,.03,'A01: A summed over r0,r1; A: summed over all ranks',
                ha='center',fontsize=8,color=MUTED)
    save(fig,out,'fig-13-10-hierarchical-allreduce.png')


def barrier(out, examples):
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    ys=[.79,.59,.39,.19]
    for stage,(ax,distance) in enumerate(zip(axes,[1,2])):
        setup(ax,f'Barrier round {stage+1}: notify distance {distance}')
        for r,y in enumerate(ys):
            box(ax,.12,y,f'r{r}',w=.15,h=.11)
            known=examples['barrier'][4][stage]['known'][r]
            label=f'r{r}: '+','.join(str(j) for j in known)
            box(ax,.84,y,label,w=.29,h=.11)
            edge(ax,(.22,y),(.67,ys[(r+distance)%4]),color=COLORS[r])
        ax.text(.12,.96,'sender',ha='center',fontsize=9)
        ax.text(.84,.96,'known arrivals after round',ha='center',fontsize=9)
        ax.text(.5,.025,'Arrows: small notifications; lists explain knowledge, not payload',
                ha='center',fontsize=8,color=MUTED)
    save(fig,out,'fig-13-9-dissemination-barrier.png')


def main():
    root=Path(__file__).resolve().parents[3]
    default=root/'drafts/Book6-推理系统导论/figures'
    if not default.exists():
        default=Path(__file__).resolve().parents[2]/'figures'
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',type=Path,default=default)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.unicode_minus':False})
    examples=build_examples()
    trees(args.out_dir)
    ring_panels(args.out_dir,examples['ring_4'],'reduce_scatter',5)
    ring_panels(args.out_dir,examples['ring_4'],'all_gather',6)
    recursive(args.out_dir)
    alltoall(args.out_dir)
    hierarchy(args.out_dir)
    barrier(args.out_dir,examples)
    print('Generated figures 13.4–13.10; integer walkthrough checks passed.')


if __name__=='__main__':
    main()

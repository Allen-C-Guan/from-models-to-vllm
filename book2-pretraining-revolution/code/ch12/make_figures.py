# Book2第12章：阅读地图与两条模型实现线；不运行实验、不覆盖正文图。
# 在选定工作目录中运行本脚本，产物写入该目录log；--help不生成图片。
import argparse
from pathlib import Path


def configure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei'):
        if name in names:
            plt.rcParams['font.family']=name
            break
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})
    return plt


def table_figure(plt,title,rows,foot,path):
    """两列教学示意；不将阅读顺序画成模型历史淘汰关系。"""
    fig,ax=plt.subplots(figsize=(12,7.5),dpi=300)
    ax.axis('off');ax.set_title(title,loc='left',fontsize=18,pad=25,color='#0b0b0b')
    table=ax.table(cellText=rows,colWidths=[.30,.70],cellLoc='left',bbox=[0,.16,1,.78])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_facecolor('#eef2f5' if col==0 else '#e3f2ed')
        cell.set_edgecolor('#a4b5bd');cell.PAD=.06
    ax.text(0,.06,foot,fontsize=9,color='#52514e',transform=ax.transAxes,va='top')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(path.with_suffix('.'+ext),facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='阅读与代码依赖示意，输出log，不训练')
    ap.add_argument('--out-dir',type=Path,default=Path.cwd()/'log/book2-ch12/figures')
    ap.add_argument('--prefix',default='reading-map')
    a=ap.parse_args()
    if Path(a.prefix).name!=a.prefix or a.prefix in ('','.','..'):raise ValueError('prefix须为文件名称')
    paths=[a.out_dir/f'{a.prefix}-12-{i}' for i in (1,2)]
    for path in paths:
        for ext in ('png','svg'):
            if path.with_suffix('.'+ext).exists():raise FileExistsError('不覆盖已有图：'+str(path))
    plt=configure();a.out_dir.mkdir(parents=True,exist_ok=True)
    table_figure(plt,'12.1 从本册蓝本继续演进',[
        ['Book1 → Book2（当前位置）','原版Transformer → 目标、可见性、分词、配方与规模'],
        ['Book3：现代组件','归一化、门控前馈、位置表示、KV缓存与头共享'],
        ['Book4：规模与效率','条件计算、状态压缩、数值格式与稳定训练'],
        ['Book5：长上下文与混合','显式历史、稀疏读取、递归状态与混合机制'],
        ['Book6 → Book7 → Book8','推理原理 → vLLM架构全景 → vLLM V1源码协作']],
        '本系列阅读依赖图；历史上并行发展的分支，不因这里的阅读顺序变成必然淘汰链。',paths[0])
    table_figure(plt,'12.2 配套实践的两条模型实现线',[
        ['ch02 / ch03','分词器实现与工具对照；BERT微调任务'],
        ['ch04 参数化GPT工厂','warmup对照；供ch08模型族、ch09探针、ch11预算实验使用'],
        ['ch08 数据与分析','固定验证窗口、模型族配置与损失统计；每个结果保留自己的协议'],
        ['ch10 三插槽GPT-2','完整Block、初始化与训练循环；ch11官方权重对拍使用此实现'],
        ['ch11 两线汇合','自训侧来自ch04工厂；官方与随机对照来自ch10；采样加载相应权重'],
        ['数学检查与图表','CPU构造例核推导；图源另存log，不替代历史训练或自动覆盖正文图']],
        '依据随书脚本的调用与加载关系。两条实现线须分别核配置、初始化、数据与评估协议。',paths[1])
    print('已生成两张示意图，各含PNG/SVG：',a.out_dir)


if __name__=='__main__':
    main()

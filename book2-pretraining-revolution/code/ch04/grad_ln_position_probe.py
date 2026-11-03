"""ch4.5 问答探针：post-LN vs pre-LN 的初始化梯度剖面（为何主路/旁路之差致命）

机制（逐站反向雅可比）：
  pre : g_l = g_{l+1}·(I + J_F·J_LN)        —— 恒等项裸露在主干上，梯度直通
  post: g_l = g_{l+1}·J_LN(s_l)·(I + J_F)   —— 恒等项被 J_LN 罩住，每站过闸
其中 J_LN 的谱尺度 ~ γ/σ(s_l)：除以该站「归一化前」的 std，且杀死共同模方向。
=> post 的梯度尺度与前向统计量倒数耦合、逐站乘法复合；pre 与前向统计解耦。

探针设计：12 层、d=384、单子层 MLP 塔（GELU，4x），隔离 LN 位置这一个变量
（注意力版即书里 6 层塔，机制同型）。初始化 N(0, w_std)，嵌入 std 可扫。
量：每站 ‖∂L/∂x_l‖（token 平均的 L2 范数，g_l/g_12 归一显示形状）与主干前向 std。
配置四档：嵌入 1 / 22.6（原典 √d 口径）/ 0.03（GPT-2 式小嵌入），及权重放大 0.1。
"""
import torch
import torch.nn.functional as F


def ln(v, gamma, beta):
    mu = v.mean(-1, keepdim=True)
    var = v.var(-1, keepdim=True, unbiased=False)
    return (v - mu) / torch.sqrt(var + 1e-5) * gamma + beta


def run(wiring, emb_std=1.0, w_std=0.02, L=12, d=384, ffr=4, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = lambda *s: (torch.randn(*s, generator=g) * w_std).requires_grad_(True)
    blocks = [dict(W1=P(d, d * ffr), b1=torch.zeros(d * ffr).requires_grad_(True),
                   W2=P(d * ffr, d), b2=torch.zeros(d).requires_grad_(True),
                   gn=torch.ones(d).requires_grad_(True),
                   bn=torch.zeros(d).requires_grad_(True)) for _ in range(L)]
    head = P(d, d)
    lnf = (torch.ones(d).requires_grad_(True), torch.zeros(d).requires_grad_(True))
    x0 = (torch.randn(1, n, d, generator=g) * emb_std).requires_grad_(True)
    tgt = torch.randn(1, n, d, generator=g)
    xs = [x0]
    x = x0
    for blk in blocks:
        x.retain_grad()
        if wiring == 'pre':
            f = F.gelu(ln(x, blk['gn'], blk['bn']) @ blk['W1'] + blk['b1']) @ blk['W2'] + blk['b2']
            x = x + f
        else:
            f = F.gelu(x @ blk['W1'] + blk['b1']) @ blk['W2'] + blk['b2']
            x = ln(x + f, blk['gn'], blk['bn'])
        xs.append(x)
    xs[-1].retain_grad()
    if wiring == 'pre':
        x = ln(x, *lnf)
    loss = ((x @ head - tgt) ** 2).mean()
    loss.backward()
    prof = [xs[l].grad[0].norm(dim=-1).mean().item() for l in range(L + 1)]
    fwd = [xs[l][0].std().item() for l in range(L + 1)]
    return prof, fwd


if __name__ == '__main__':
    print('每站输入梯度 g_l（以 g_12=1.00 归一；12 层 MLP 塔，d=384，seed=0）')
    print(f"{'l =':>6}" + ''.join(f'{l:>8d}' for l in range(13)))
    configs = [(1.0, 0.02), (22.6, 0.02), (0.03, 0.02), (1.0, 0.1)]
    for wiring in ['pre', 'post']:
        for emb_std, w_std in configs:
            prof, fwd = run(wiring, emb_std, w_std)
            tag = f'{wiring} emb={emb_std:<5} w={w_std}'
            print(f'{tag:>26}' + ''.join(f'{v / prof[-1]:>8.2f}' for v in prof))
            print(f'{"  (主干std)":>26}' + ''.join(f'{v:>8.2f}' for v in fwd))

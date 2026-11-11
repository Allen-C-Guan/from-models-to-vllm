"""用途：比较post/pre-LN初始化梯度剖面的小型MLP探针；不是训练稳定性实验。
所属章节：Book2第4章4.5节；运行：在本册目录 python code/ch04/grad_ln_position_probe.py。

采用行向量梯度约定：g_l = g_{l+1} J，故
  pre : J = I + J_F(LN(x)) J_LN(x)
  post: J = J_LN(x+F(x)) (I + J_F(x))
pre中的局部I绕过LN，但分支及末端LN仍依赖前向统计，不能称为完全解耦。
这里是12层、d=384、无注意力的MLP塔，损失为随机目标的均方误差；
pre配置还加末端LN，所以比较的是两套LN配置，并非只移动一处算子。
输出各位置梯度L2范数的均值，以及整张激活的样本标准差。
22.6保留为历史大尺度构造值，不是sqrt(384)，也不代表完整原典嵌入初始化。
"""
import torch
import torch.nn.functional as F


def ln(v, gamma, beta):
    mu = v.mean(-1, keepdim=True)
    var = v.var(-1, keepdim=True, unbiased=False)
    return (v - mu) / torch.sqrt(var + 1e-5) * gamma + beta


def run(wiring, emb_std=1.0, w_std=0.02, L=12, d=384, ffr=4, n=64, seed=0):
    if wiring not in ("pre", "post"):
        raise ValueError("wiring须为pre或post")
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
